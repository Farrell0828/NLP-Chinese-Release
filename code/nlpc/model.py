import math
import torch
from torch import nn
from torch.nn import Module
from transformers import AutoModel


class PretrainedTransformer(Module):
    r"""
    The wrapper for those pretrained transformer models support Chinese
    provided by Huggingface transformer libary.

    Args:
        config (Dict): The model configration dictionary at least with key
            `pretrained_model_name`, which can be either:

                - A string with the `shortcut name` of a pretrained model to
                  load from cache or download, e.g., ``bert-base-chinese``.
                - A string with the `identifier name` of a pretrained model
                  that was user-uploaded to Huggingface's S3, e.g.,
                    ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.,
                  ``./my_model_directory/``.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_model_name = config['pretrained_model_name']
        self.model = AutoModel.from_pretrained(config['pretrained_model_name'])

    def forward(self, batch):
        r"""
        Forward pass.

        Args:
            batch (Dict): The input batch. A Dictionary with at least keys
                `input_ids`, `attention_mask` for all kind of models and
                `token_type_ids` for some models.

        Returns:
            (:class:`~transformers.file_utils.ModelOutput`): The dictionary-
                like model outputs with keys `last_hidden_state`,
                `hidden_states`, `attentions`, `length` for all kind of
                models.
        """
        if 'distilbert' in self.pretrained_model_name:
            ret = self.model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )
        elif 'xlm-mlm' in self.pretrained_model_name:
            ret = self.model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                langs=self.model.config.lang2id['zh'],
                token_type_ids=batch['token_type_ids'],
                output_hidden_states=True,
                return_dict=True
            )
        else:
            ret = self.model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids'],
                output_hidden_states=True,
                return_dict=True
            )
        ret['length'] = batch['length']
        ret['attention_mask'] = batch['attention_mask']
        return ret


class PoolerMultiHeadAttention(Module):
    r"""
    Implement the Multi-Head Attention module used in the pooler layer.
    """
    def __init__(self, config):
        super().__init__()
        if config['hidden_size'] % config['pooler_num_attention_heads'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of "
                "attention heads (%d)" % (
                    config['hidden_size'], config['pooler_num_attention_heads']
                )
            )
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['pooler_num_attention_heads']
        self.attention_head_size = int(
            config['hidden_size'] / config['pooler_num_attention_heads']
        )
        self.query_fc = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key_fc = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.value_fc = nn.Linear(config['hidden_size'], config['hidden_size'])

    def transpose_for_scores(self, x):
        r"""
        Transpose the input tnesor before cumpute attention scores.

        Args:
            x (Tensor): Input tensor withs shape (batch_size,
                sequence_length, hidden_size).

        Returns:
            (Tensor): Transposed tensor with shape (batch_size,
                num_attention_heads, sequence_length, attention_head_size)
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask=None):
        r"""
        Args:
            query (Tensor): Query tensor with shape (batch_size,
                sequence_length, hidden_size).
            key (Tensor): Key tensor with the same shape of query.
            value (Tensor): Value tensor with the same shape of query.
            attention_mask (Tensor, optional): Mask to avoid performing
                attention on the padding token indices. (Default `None`, which
                means that no padding token need to be masked.)

        Returns:
            (Tensor): Summary tensor with shape (batch_size, sequence_length,
                hidden_size).
            """
        mixed_query = self.query_fc(query)
        mixed_key = self.key_fc(key)
        mixed_value = self.value_fc(value)

        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = (
            attention_scores / math.sqrt(self.attention_head_size)
        )
        if attention_mask is not None:
            attention_scores = (
                attention_scores - ~attention_mask * 1e12
            )
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        content = torch.matmul(attention_probs, value)
        content = content.permute(0, 2, 1, 3).contiguous()
        new_content_shape = content.size()[:-2] + (self.hidden_size, )
        content = content.view(*new_content_shape)
        return content


class MixtureOfLayers(Module):
    r"""
    Mixture of Layers module. This module take transformer's all layers
    hidden states and task type index as inputs and mix different layers'
    hidden states together as output depend on task type.
    """
    def __init__(self, config):
        super().__init__()
        self.task_type_embedding = nn.Embedding(
            len(config['targets_num_classes']),
            config['num_transformer_layers']
        )

    def forward(self, hidden_states, task_type_id):
        r"""
        Args:
            hidden_states (Tuple[Tensor]): Transformer's all layer hidden
                states. Every element in this tuple is a tensor with shape
                (batch_size, sequence_length, hidden_size).
            task_type_id (Tensor): Tensor with shape (batch_size, ).
        """
        # (batch_size, num_transformer_layers)
        layer_score = self.task_type_embedding(task_type_id)

        # (batch_size, 1, num_transformeer_layers)
        layer_prob = nn.Softmax(dim=-1)(layer_score).unsqueeze(-2)

        # (batch_size, num_transformer_layers, hidden_size)
        first_hidden_states = torch.cat([
            hidden_state[:, 0, :].unsqueeze(-2)
            for hidden_state in hidden_states[1:]
        ], dim=-2)

        # (batch_size, hidden_size)
        output_feature = torch.matmul(
            layer_prob, first_hidden_states
        ).squeeze(-2)

        return output_feature


class Pooler(Module):
    r"""
    The pooling layer support multiple pooling method:

        - `origin`: The original pooler output which used in text pretrain
            classification task (like BERT's next sentence prediction task).
        - `max`: Apply max pooling along the sequence dimension.
        - `mean`: Apply average pooling along the sequence dimension.
        - `first`: Simply take the first token hidden state (like BERT).
        - `last`: Simply take the last token hidden state (like XLNet).
        - `attn`: Use multi-head attention to summary the sequence.
        - `mol`: Mixture of Layers. Mix the different transformer layer's
            first token outputs.
    """
    def __init__(self, config):
        super().__init__()
        self.pooling_method = config['pooling_method']
        if config['pooling_method'] == 'origin':
            pass
        elif config['pooling_method'] == 'max':
            self.pooler = lambda x: torch.max(x, dim=1)[0]
        elif config['pooling_method'] == 'mean':
            self.pooler = lambda x: torch.mean(x, dim=1)
        elif config['pooling_method'] == 'first':
            self.pooler = lambda x: x[:, 0]
        elif config['pooling_method'] == 'last':
            self.pooler = lambda x: x[:, -1]
        elif config['pooling_method'] == 'attn':
            self.pooler = PoolerMultiHeadAttention(config)
            self.task_type_embedding = nn.Embedding(
                len(config['targets_num_classes']), config['hidden_size']
            )
        elif config['pooling_method'] == 'mol':
            self.pooler = MixtureOfLayers(config)
        elif config['pooling_method'] == 'inter':
            self.pooler = lambda x: x[config['inter_num']][:, 0]
        else:
            raise ValueError(
                'pooling_method {} not support now.'
                .format(config['pooling_method'])
            )

    def forward(self, transformer_output_dict, task_type_id=None):
        if self.pooling_method == 'origin':
            return transformer_output_dict['pooler_output']
        if self.pooling_method == 'attn':
            task_type = self.task_type_embedding(task_type_id)
            return self.pooler(
                qurey=task_type,
                key=transformer_output_dict['last_hidden_state'],
                value=transformer_output_dict['last_hidden_state'],
                attention_mask=transformer_output_dict['attention_mask']
            )
        elif self.pooling_method == 'mol':
            return self.pooler(
                hidden_states=transformer_output_dict['hidden_states'],
                task_type_id=task_type_id
            )
        elif self.pooling_method == 'inter':
            return self.pooler(transformer_output_dict['hidden_states'])
        else:
            return self.pooler(transformer_output_dict['last_hidden_state'])


class SingleHeadPredictor(Module):
    r"""
    Implement `single_head` parameter share architecture. To single model.
    """
    def __init__(self, config):
        super().__init__()
        task_name = config['share_architecture']
        self.predict_heads = nn.ModuleDict()
        self.predict_heads[task_name] = nn.Linear(
            config['hidden_size'],
            config['targets_num_classes'][task_name]
        )

    def forward(self, pooled_features):
        r"""
        Args:
            pooled_features (Tensor): Input tensor with shape (batch_size,
                hidden_size).

            Returns:
                (Tensor): (batch_size, num_classes_of_this_task).
        """
        logits = {}
        for task_name, linear_layer in self.predict_heads.items():
            logits[task_name] = linear_layer(pooled_features)
        return logits


class HardMultiHeadPredictor(Module):
    r"""
    Implement `hard_multi_head` parameter share architecture. Each head in
    this predictor corresponding to one task.
    """
    def __init__(self, config):
        super().__init__()
        self.predict_heads = nn.ModuleDict()
        for task_name, num_classes in config['targets_num_classes'].items():
            self.predict_heads[task_name] = nn.Linear(
                config['hidden_size'], num_classes
            )

    def forward(self, pooled_features):
        r"""
        Args:
            pooled_features (Tensor): Input tensor with shape (batch_size,
                hidden_size).

            Returns:
                (Dict[str, Tensor]): Output a dictionary with task name as
                    key and corresponding logits, which has the shape of
                    (batch_size, num_classes_of_this_task), as value.
        """
        logits = {}
        for task_name, linear_layer in self.predict_heads.items():
            logits[task_name] = linear_layer(pooled_features)
        return logits


class HardPlusMultiHeadPredictor(Module):
    r"""
    Implement `head_plus` parameter share architecture. Each head in
    this predictor corresponding to one task.
    """
    def __init__(self, config):
        super().__init__()
        self.inter_linears = nn.ModuleDict()
        self.predict_heads = nn.ModuleDict()
        self.activation = nn.Tanh()
        for task_name, num_classes in config['targets_num_classes'].items():
            self.inter_linears[task_name] = nn.Linear(
                config['hidden_size'], config['hidden_size']
            )
            self.predict_heads[task_name] = nn.Linear(
                config['hidden_size'], num_classes
            )

    def forward(self, pooled_features):
        r"""
        Args:
            pooled_features (Tensor): Input tensor with shape (batch_size,
                hidden_size).

            Returns:
                (Dict[str, Tensor]): Output a dictionary with task name as
                    key and corresponding logits, which has the shape of
                    (batch_size, num_classes_of_this_task), as value.
        """
        logits = {}
        for task_name in self.inter_linears:
            inter_output = self.inter_linears[task_name](pooled_features)
            inter_output = self.activation(inter_output)
            logits[task_name] = self.predict_heads[task_name](inter_output)
        return logits


class Expert(Module):
    r"""
    One expert is actually multi-layer perceptrons with ReLU activations.
    """
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            nn.Linear(config['hidden_size'], config['expert_hidden_size'])
        )
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=config['expert_dropout']))

        # Intermediate layers
        for _ in range(config['expert_num_hidden_layers'] - 1):
            self.layers.append(
                nn.Linear(
                    config['expert_hidden_size'], config['expert_hidden_size']
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=config['expert_dropout']))

        # Output layer
        self.layers.append(
            nn.Linear(config['expert_hidden_size'], config['hidden_size'])
        )

    def forward(self, x):
        r"""
        Args:
            x (Tensor): Input feature with shape (batch_size, hidden_size).

        Returns:
            (Tensor): Output feature with shape (batch_size, hidden_size).
        """
        for layer in self.layers:
            x = layer(x)
        return x


class MixtureOfExperts(Module):
    r"""
    Mixture of Experts layer.
    """
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config['num_experts'])
        ])
        self.gate_linear = nn.Linear(
            config['hidden_size'], config['num_experts']
        )

    def forward(self, input_feature):
        r"""
        Args:
            input_feature (Tensor): Input feature with shape (batch_size,
                hidden_size).

        Returns:
            (Tensor): Output feature with shape (batch_size, hidden_size).
        """
        experts_outputs = torch.cat(
            [expert(input_feature).unsqueeze(-2) for expert in self.experts],
            dim=-2
        )
        experts_scores = self.gate_linear(input_feature)
        experts_probs = nn.Softmax(dim=-1)(experts_scores).unsqueeze(-2)
        output_feature = torch.matmul(experts_probs, experts_outputs)
        return output_feature.squeeze(-2)


class MixtureOfExpertsPreditor(Module):
    r"""
    Implement `moe` (MoE, Mixture of Experts) parameter share architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.moe_layer = MixtureOfExperts(config)
        self.hard_preditor = HardMultiHeadPredictor(config)

    def forward(self, pooled_features):
        return self.hard_preditor(self.moe_layer(pooled_features))


class MultiGateMixtureOfExperts(Module):
    r"""
    Multi-gate Mixture of Experts layer.
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config['num_experts']
        self.num_tasks = len(config['targets_num_classes'])
        self.hidden_size = config['hidden_size']
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config['num_experts'])
        ])
        self.gates_linear = nn.Linear(
            config['hidden_size'], self.num_experts * self.num_tasks
        )

    def forward(self, input_feature, task_type_id):
        r"""
        Args:
            input_feature (Tensor): Input feature with shape (batch_size,
                hidden_size).
            task_type_id (Tensor): Tensor with shape (batch_size, )

        Returns:
            (Tensor): Output feature with shape (batch_size, hidden_size).
        """
        experts_outputs = torch.cat(
            [expert(input_feature).unsqueeze(-2) for expert in self.experts],
            dim=-2
        ).unsqueeze(-3)
        experts_scores = self.gates_linear(input_feature)
        new_socres_shape = experts_scores.size()[:-1] + (
            self.num_tasks, self.num_experts
        )
        experts_scores = experts_scores.view(*new_socres_shape)
        experts_probs = nn.Softmax(dim=-1)(experts_scores).unsqueeze(-2)
        output_feature = torch.matmul(experts_probs, experts_outputs)
        output_feature = output_feature.squeeze(-2)
        task_type_id = (
            task_type_id[:, None, None].expand(-1, -1, self.hidden_size)
        ).long()
        gathered_feature = output_feature.gather(1, task_type_id).squeeze(1)
        return gathered_feature


class MultiGateMixtureOfExpertsPredictor(Module):
    r"""
    Implement `mmoe` (MMoE, Multi-gate Mixture of Experts) parameter share
    architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.mmoe_layer = MultiGateMixtureOfExperts(config)
        self.hard_preditor = HardMultiHeadPredictor(config)

    def forward(self, pooled_features, task_type_id):
        gathered_features = self.mmoe_layer(pooled_features, task_type_id)
        return self.hard_preditor(gathered_features)


class NLPCSingleHeadPredictModel(Module):
    r"""
    The complete single-tasks model for NLP Chineae Comarchitapetetion with
    single-head prediction architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_transformer = PretrainedTransformer(config)
        self.pooler = Pooler(config)
        self.predictor = SingleHeadPredictor(config)

    def forward(self, batch):
        transformer_outputs = self.pretrained_transformer(batch)
        pooled_features = self.pooler(
            transformer_outputs,
            task_type_id=batch['task_type_id'].long()
        )
        logits = self.predictor(pooled_features)
        return logits


class NLPCHardMultiHeadPredictModel(Module):
    r"""
    The complete multi-tasks model for NLP Chineae Comarchitapetetion with hard
    share multi-head prediction architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_transformer = PretrainedTransformer(config)
        self.pooler = Pooler(config)
        self.predictor = HardMultiHeadPredictor(config)

    def forward(self, batch):
        transformer_outputs = self.pretrained_transformer(batch)
        pooled_features = self.pooler(
            transformer_outputs,
            task_type_id=batch['task_type_id'].long()
        )
        logits = self.predictor(pooled_features)
        return logits


class NLPCHardPlusMultiHeadPredictModel(Module):
    r"""
    The complete multi-tasks model for NLP Chineae Comarchitapetetion with hard
    share multi-head prediction architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_transformer = PretrainedTransformer(config)
        self.pooler = Pooler(config)
        self.predictor = HardPlusMultiHeadPredictor(config)

    def forward(self, batch):
        transformer_outputs = self.pretrained_transformer(batch)
        pooled_features = self.pooler(
            transformer_outputs,
            task_type_id=batch['task_type_id'].long()
        )
        logits = self.predictor(pooled_features)
        return logits


class NLPCMoEPredictModel(Module):
    r"""
    The complete multi-tasks model for NLP Chineae Comarchitapetetion with
    Mixture of Experts prediction architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_transformer = PretrainedTransformer(config)
        self.pooler = Pooler(config)
        self.predictor = MixtureOfExpertsPreditor(config)

    def forward(self, batch):
        transformer_outputs = self.pretrained_transformer(batch)
        pooled_features = self.pooler(
            transformer_outputs,
            task_type_id=batch['task_type_id'].long()
        )
        logits = self.predictor(pooled_features)
        return logits


class NLPCMMoEPredictModel(Module):
    r"""
    The complete multi-tasks model for NLP Chineae Comarchitapetetion with
    Multi-gate Mixture of Experts prediction architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.pretrained_transformer = PretrainedTransformer(config)
        self.pooler = Pooler(config)
        self.predictor = MultiGateMixtureOfExpertsPredictor(config)

    def forward(self, batch):
        transformer_outputs = self.pretrained_transformer(batch)
        pooled_features = self.pooler(
            transformer_outputs,
            task_type_id=batch['task_type_id'].long()
        )
        logits = self.predictor(pooled_features, batch['task_type_id'])
        return logits


class NLPCModel(Module):
    r"""
    Convenience wrapper module for different parameters share architecture
    models.
    """
    def __init__(self, config):
        super().__init__()
        if config['share_architecture'] == 'hard':
            self.model = NLPCHardMultiHeadPredictModel(config)
        elif config['share_architecture'] == 'hard_plus':
            self.model = NLPCHardPlusMultiHeadPredictModel(config)
        elif config['share_architecture'] == 'moe':
            self.model = NLPCMoEPredictModel(config)
        elif config['share_architecture'] == 'mmoe':
            self.model = NLPCMMoEPredictModel(config)
        elif config['share_architecture'] in [
            'ocnli', 'ocemotion', 'tnews', 'cmnli'
        ]:
            self.model = NLPCSingleHeadPredictModel(config)
        # elif config['share_architecture'] == 'ocnli':
        #     self.model = NLPCSingleHeadPredictModel(config, 'ocnli')
        # elif config['share_architecture'] == 'ocemotion':
        #     self.model = NLPCSingleHeadPredictModel(config, 'ocemotion')
        # elif config['share_architecture'] == 'tnews':
        #     self.model = NLPCSingleHeadPredictModel(config, 'tnews')
        else:
            raise ValueError(
                'share_architecture {} not support now.'
                .format(config['share_architecture'])
            )

    def forward(self, batch):
        return self.model(batch)
