import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer


TASK_NAME2INT = {
    'ocnli': 0, 'ocemotion': 1, 'tnews': 2
}

OCNLI_NAME2INT = {
    'entailment': 0, 'neutral': 1, 'contradiction': 2
}

OCEMOTION_TARGET2INT = {
    'sadness': 0, 'happiness': 1, 'disgust': 2,
    'like': 3, 'anger': 4, 'surprise': 5, 'fear': 6
}

TNEWS_TARGET2INT = {
    109: 0, 104: 1, 102: 2, 113: 3, 107: 4, 101: 5, 103: 6, 110: 7,
    108: 8, 112: 9, 116: 10, 115: 11, 106: 12, 100: 13, 114: 14
}


class NLPCBaseDataset(Dataset):
    r"""
    The base Dataset calss for NLP Chinese Competition.

    Args:
        config: (Dict): Configuration releted to the dataset.
        csv_path (str): Path to the dataset .csv file.
        id_col_name (str): Name of the id column. Should be one of the
            `col_names`
        first_text_col_name (str): Name of the text column. Should be one
            of the `col_names`.
        task_name (str): Name of the task this dataset belong to.
        col_names (List[str], optional): Names of the :class:`pandas.DataFrame`
            columns. If not provided, will infer the column names from the
            data. (Default None).
        sep (str, optional): Delimiter to use in the `pandas.read_csv`
            function. (Default `,`)
        split (str, optional): Which split this dataset belong to.
            `train`, `val`, `trainval` or `test`. (Default: `train`)
        n_folds (int, optional): Number of folds. (Default 1, which means that
            all the dataset will be used).
        fold (int, optional): Only work when n_folds > 1. Which fold this
            dataset belong to. Should in the range [0, n_folds). (Default 0)
        overfit (bool, optional): Only sample 4 samples if set to `True` so
            that model could overfit on them. Meaning for debugging. Default
            `Fales`.
        second_text_col_name (str, optional): Name of the second text
            column if existed. `None` or one of the `col_names`. Default
            `None`, which means that no second text existed.
        target_col_name (str): Name of target columns. It should be `None`
            for `test` split cause there is no target avaiable and one of
            the `col_names` for `train` and `val` split.
        target2int (Dict[Any: int], optional): The dict used to map target
            values to inteters. Default `None`, which means that there is
            no need to map target values to integers. It should be `None`
            for `test` split.
        tensor_type (str, optional): Default `None`, which means that
            return list of python integers. If set, will return tensors.
            Acceptable values are:
            * 'pt': Return PyTorch `torch.Tensor` objects.
            * 'np': Return Numpy `np.ndarray` objects.
    """
    def __init__(self, config, csv_path, id_col_name,
                 first_text_col_name, task_name,
                 col_names=None,
                 sep=',',
                 split='train',
                 n_folds=1,
                 fold=0,
                 overfit=False,
                 second_text_col_name=None,
                 target_col_name=None,
                 target2int=None,
                 tensor_type=None):
        super().__init__()
        df = pd.read_csv(
            csv_path,
            names=col_names,
            sep=sep,
            encoding='utf-8',
            engine='python'
        )
        if target2int:
            df[target_col_name] = (
                df[target_col_name].map(target2int)
            )

        if split in ['trainval', 'test'] or n_folds < 2:
            self.df = df
        else:
            kf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=606
            )
            train_indexes, val_indexes = list(
                kf.split(df.index, y=df[target_col_name].astype('category'))
            )[fold]
            self.df = (
                df.iloc[train_indexes]
                if split == 'train'
                else df.iloc[val_indexes]
            )
            self.df = self.df.reset_index(drop=True)

        if overfit:
            self.df = self.df.iloc[:4, :]

        self.id_col_name = id_col_name
        self.first_text_col_name = first_text_col_name
        self.task_name = task_name
        self.second_text_col_name = second_text_col_name
        self.target_col_name = target_col_name
        self.split = split
        self.target2int = target2int
        self.tensor_type = tensor_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_model_name']
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item_id = self.df[self.id_col_name][index]
        first_text = self.df.loc[index, self.first_text_col_name]
        second_text = (
            self.df.loc[index, self.second_text_col_name]
            if self.second_text_col_name
            else None
        )

        tokenize_results = self.tokenizer(
            text=first_text,
            text_pair=second_text,
            add_special_tokens=True,
            return_tensors=None,
            return_token_type_ids=True,
            return_length=True
        )

        ret = {
            'id': item_id,
            'input_ids': tokenize_results['input_ids'],
            'token_type_ids': tokenize_results['token_type_ids'],
            'length': tokenize_results['length']
        }

        if self.split in ['train', 'val', 'trainval']:
            target = self.df.loc[index, self.target_col_name]
            ret['target'] = target

        task_type_id = TASK_NAME2INT[self.task_name]
        ret['task_type_id'] = task_type_id

        if self.tensor_type is None:
            pass
        elif self.tensor_type == 'np':
            for k, v in ret.items():
                ret[k] = np.array(v)
        elif self.tensor_type == 'pt':
            for k, v in ret.items():
                ret[k] = torch.tensor(v)
        else:
            raise ValueError(
                f'tenosr_type {self.tensor_type} not support now.'
            )

        return ret


class OCNLIDataset(NLPCBaseDataset):
    """
    Inherented from :class:`NLPCBaseDataset` class. The dataset of NLP
    Chinese Competetion Task 1: OCNLI.

    Args:
        config: (Dict): Configuration releted to the dataset.
        split (str, optional): Which split this dataset belong to.
            `train`, `val`, `trainval` or `test`. (Default: `train`)
        overfit (bool, optional): Only sample 4 samples if set to `True` so
            that model could overfit on them. Meaning for debugging. Default
            `Fales`.
        tensor_type (str, optional): Default `None`, which means that
            return list of python integers. If set, will return tensors.
            Acceptable values are:
            * 'pt': Return PyTorch `torch.Tensor` objects.
            * 'np': Return Numpy `np.ndarray` objects.
    """
    def __init__(self, config,
                 split='train',
                 overfit=False,
                 tensor_type=None):
        if split in ['train', 'val']:
            if 'ocnli_' + split + '_path' not in config:
                csv_path = config['ocnli_trainval_path']
                col_names = ['id', 'sentence1', 'sentence2', 'label']
                sep = '\\t'
                n_folds = 5
                target2int = None
            else:
                csv_path = config['ocnli_' + split + '_path']
                col_names = None
                sep = ','
                n_folds = 1
                target2int = OCNLI_NAME2INT
            target_col_name = 'label'
        elif split == 'trainval':
            csv_path = config['ocnli_trainval_path']
            col_names = ['id', 'sentence1', 'sentence2', 'label']
            sep = '\\t'
            n_folds = 1
            target2int = None
            target_col_name = 'label'
        elif split == 'test':
            csv_path = config['ocnli_test_path']
            col_names = ['id', 'sentence1', 'sentence2']
            target_col_name = None
            sep = '\\t'
            n_folds = 1
            target2int = None
        else:
            raise ValueError()

        super().__init__(
            config=config,
            csv_path=csv_path,
            col_names=col_names,
            sep=sep,
            first_text_col_name='sentence1',
            id_col_name='id',
            task_name='ocnli',
            split=split,
            n_folds=n_folds,
            overfit=overfit,
            second_text_col_name='sentence2',
            target_col_name=target_col_name,
            target2int=target2int,
            tensor_type=tensor_type
        )


class OCEMOTIONDataset(NLPCBaseDataset):
    r"""
    Inherented from :class:`NLPCBaseDataset` class. The dataset of NLP
    Chinese Competetion Task2: OCEMOTION.

    Args:
        config: (Dict): Configuration releted to the dataset.
        split (str, optional): Which split this dataset belong to.
            `train`, `val` `trainval` or `test`. (Default: `train`)
        overfit (bool, optional): Only sample 4 samples if set to `True` so
            that model could overfit on them. Meaning for debugging. Default
            `Fales`.
        tensor_type (str, optional): Default `None`, which means that
            return list of python integers. If set, will return tensors.
            Acceptable values are:
            * 'pt': Return PyTorch `torch.Tensor` objects.
            * 'np': Return Numpy `np.ndarray` objects.
    """
    def __init__(self, config,
                 split='trian',
                 overfit=False,
                 tensor_type=None):
        if split == 'train' and 'ocemotion_train_path' in config:
            csv_path = config['ocemotion_train_path']
            col_names = None
            sep = ','
            n_folds = 1
            target_col_name = 'label'
            target2int = OCEMOTION_TARGET2INT
        else:
            col_names = ['id', 'sentence']
            sep = '\\t'
            if split in ['train', 'val', 'trainval']:
                csv_path = config['ocemotion_trainval_path']
                col_names.append('label')
                target_col_name = 'label'
                n_folds = 1 if split == 'trainval' else 5
                target2int = OCEMOTION_TARGET2INT
            elif split == 'test':
                csv_path = config['ocemotion_test_path']
                target_col_name = None
                target2int = None
                n_folds = 1
            else:
                raise ValueError()

        super().__init__(
            config=config,
            csv_path=csv_path,
            col_names=col_names,
            sep=sep,
            id_col_name='id',
            first_text_col_name='sentence',
            task_name='ocemotion',
            split=split,
            n_folds=n_folds,
            overfit=overfit,
            target_col_name=target_col_name,
            target2int=target2int,
            tensor_type=tensor_type
        )


class TNEWSDataset(NLPCBaseDataset):
    r"""
    Inherented from :class:`NLPCBaseDataset` class. The dataset of NLP
    Chinese Competetion Task3: TNEWS.

    Args:
        config: (Dict): Configuration releted to the dataset.
        split (str, optional): Which split this dataset belong to.
            `train`, `val` `trainval` or `test`. (Default: `train`)
        overfit (bool, optional): Only sample 4 samples if set to `True` so
            that model could overfit on them. Meaning for debugging. Default
            `Fales`.
        tensor_type (str, optional): Default `None`, which means that
            return list of python integers. If set, will return tensors.
            Acceptable values are:
            * 'pt': Return PyTorch `torch.Tensor` objects.
            * 'np': Return Numpy `np.ndarray` objects.
    """
    def __init__(self, config,
                 split='trian',
                 overfit=False,
                 tensor_type=None):
        if split in ['train', 'val']:
            if 'tnews_' + split + '_path' not in config:
                csv_path = config['tnews_trainval_path']
                col_names = ['id', 'sentence', 'label']
                sep = '\\t'
                n_folds = 5
            else:
                csv_path = config['tnews_' + split + '_path']
                col_names = None
                sep = ','
                n_folds = 1
            target_col_name = 'label'
            target2int = TNEWS_TARGET2INT
        elif split == 'trainval':
            csv_path = config['tnews_trainval_path']
            col_names = ['id', 'sentence', 'label']
            sep = '\\t'
            n_folds = 1
            target_col_name = 'label'
            target2int = TNEWS_TARGET2INT
        elif split == 'test':
            csv_path = config['tnews_test_path']
            col_names = ['id', 'sentence']
            sep = '\\t'
            n_folds = 1
            target_col_name = None
            target2int = None
        else:
            raise ValueError()

        super().__init__(
            config=config,
            csv_path=csv_path,
            col_names=col_names,
            sep=sep,
            id_col_name='id',
            first_text_col_name='sentence',
            task_name='tnews',
            split=split,
            n_folds=n_folds,
            overfit=overfit,
            target_col_name=target_col_name,
            target2int=target2int,
            tensor_type=tensor_type
        )


class NLPCJointDataset(Dataset):
    r"""
    The dataset of NLP Chinese Competetion joint tasks.

    Args:
        config: (Dict): Configuration releted to the dataset.
        split (str, optional): Which split this dataset belong to.
            `train`, `val` or `test`. (Default: `train`)
        oversample(bool, optional): Wether or not over sample samll number
            task dataset. (Default `False`).
        n_folds (int, optional): Number of folds. (Default 1, which means that
            all the dataset will be used).
        fold (int, optional): Only work when n_folds > 1. Which fold this
            dataset belong to. Should in the range [0, n_folds). (Default 0)
        overfit (bool, optional): Only sample 4 samples if set to `True` so
            that model could overfit on them. Meaning for debugging. Default
            `Fales`.
        tensor_type (str, optional): Default `None`, which means that
            return list of python integers. If set, will return tensors.
            Acceptable values are:
            * 'pt': Return PyTorch `torch.Tensor` objects.
            * 'np': Return Numpy `np.ndarray` objects.
    """
    def __init__(self, config,
                 split='train',
                 oversample=False,
                 overfit=False,
                 tensor_type=None):
        super().__init__()
        assert(split in ['train', 'val', 'trainval', 'test'])
        self.split = split
        self.oversample = oversample
        self.ocnli_dataset = OCNLIDataset(
            config=config,
            split=split,
            overfit=overfit,
            tensor_type=tensor_type
        )
        self.ocemotion_dataset = OCEMOTIONDataset(
            config=config,
            split=split,
            overfit=overfit,
            tensor_type=tensor_type
        )
        self.tnews_dataset = TNEWSDataset(
            config=config,
            split=split,
            overfit=overfit,
            tensor_type=tensor_type
        )
        self.ocnli_len = len(self.ocnli_dataset)
        self.ocemotion_len = len(self.ocemotion_dataset)
        self.tnews_len = len(self.tnews_dataset)
        self.sum_len = sum(
            (self.ocnli_len, self.ocemotion_len, self.tnews_len)
        )
        self.max_len = max(
            (self.ocnli_len, self.ocemotion_len, self.tnews_len)
        )

    def __len__(self):
        return (
            3 * self.max_len
            if self.split == 'train' and self.oversample
            else self.sum_len
        )

    def __getitem__(self, index):
        if 0 <= index < self.ocnli_len:
            return self.ocnli_dataset[index]
        elif index < self.ocnli_len + self.ocemotion_len:
            return self.ocemotion_dataset[index-self.ocnli_len]
        elif index < self.sum_len:
            return self.tnews_dataset[index-self.ocnli_len-self.ocemotion_len]
        elif index < self.sum_len + (self.max_len - self.ocnli_len):
            return self.ocnli_dataset[np.random.randint(self.ocnli_len)]
        elif index < self.sum_len + (
            self.max_len - self.ocnli_len
        ) + (self.max_len - self.ocemotion_len):
            return self.ocemotion_dataset[
                np.random.randint(self.ocemotion_len)
            ]
        elif index < self.__len__():
            return self.tnews_dataset[np.random.randint(self.tnews_len)]
        else:
            raise IndexError()


def pad_seq(encoded_inputs, max_length,
            pad_token_id=0,
            return_attention_mask=True):
    r"""
    Padding input sequence to max sequence length with `pad_token_id` and
    update other infomation if needed. Support both `List` and `np.ndarray`
    input sequence.

    Args:
        encoded_inputs (Dict): Dictionary of tokenized inputs (`List[int]` or
            `np.ndarray`) with 'input_ids' as key and additional information.
        max_length: maximum length of the returned list.
        pad_token_id (int): The id of pad token in the vocabulary. May
            specified by model. (Default 0)
        return_attention_mask (bool, optional): Set to False to avoid
            returning attention mask. (default: True)

    Returns:
        (Dict): Updated `encoded_inputs` with padded input_ids and attention
            mask if `return_attentioin_mask` if True.
    """
    origin_length = len(encoded_inputs["input_ids"])
    difference = max_length - origin_length
    if isinstance(encoded_inputs['input_ids'], list):
        if return_attention_mask:
            encoded_inputs["attention_mask"] = (
                [1] * origin_length + [0] * difference
            )
        if "token_type_ids" in encoded_inputs:
            encoded_inputs["token_type_ids"] = (
                encoded_inputs["token_type_ids"] + [0] * difference
            )
        encoded_inputs["input_ids"] = (
            encoded_inputs["input_ids"] + [pad_token_id] * difference
        )
    elif isinstance(encoded_inputs['input_ids'], np.ndarray):
        if return_attention_mask:
            attention_mask = np.zeros(max_length)
            attention_mask[:origin_length] = 1
            encoded_inputs["attention_mask"] = attention_mask
        if "token_type_ids" in encoded_inputs:
            token_type_ids = np.zeros(max_length).astype(np.int64)
            token_type_ids[:origin_length] = encoded_inputs['token_type_ids']
            encoded_inputs["token_type_ids"] = token_type_ids
        input_ids = np.full(max_length, pad_token_id).astype(np.int64)
        input_ids[:origin_length] = encoded_inputs['input_ids']
        encoded_inputs["input_ids"] = input_ids
    return encoded_inputs


def collate_fn_with_padding(batch):
    r"""
    Padding every sample in a list of samples to max length of these samples
    and merge them to form a mini-batch of Tensor(s). Each sample is an
    encoded inputs. This function padding each sample's input_ids to max
    length and update other information if needed. Then, for each item in a
    sample, puts each data field into a tensor with outer dimension batch size.

    Args:
        batch (List[Dict]): List of samples.

    Returns:
        (Dict): Padded and merged batch.
    """
    max_len = max([sample['length'] for sample in batch])
    padded_batch = [pad_seq(sample, max_len) for sample in batch]
    return default_collate(padded_batch)
