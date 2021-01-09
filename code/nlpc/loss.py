import math
import torch
from torch import nn
from torch.nn import functional as F

from .metric import F1ScoreMetric


class SoftF1Loss(nn.Module):
    r"""
    Differentiable loss function directly optimize F1 Score for classification
    task. In the multi-class case this is the average of the soft F1 loss of
    each class with weighting depending on the `average` parameter.

    Args:
        average (str, optional): One of ['micro', 'macro']. This parameter
            determines the type of averaging performed on the data:

            - `micro`: Calculate loss globally by counting the total true
                positives, false negatives and false positives.
            - `macro`: Calculate loss for each label, and find their unweighted
                mean. This does not take label imbalance into account.

        eps (float): Epsilon value to avoid zero division.
    """
    def __init__(self, average='macro', eps=1e-7):
        super().__init__()
        if average not in ['macro', 'micro']:
            raise ValueError('average should be "macro" or "micro".')
        self.average = average
        self.eps = eps

    def forward(self, logit, target):
        r"""
        Args:
            logit (Tensor): Classification model's output before softmax.
                Tensor with shape (batch_size, num_classes_of_this_task).
            target (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).
        """
        y_pred = F.softmax(logit, dim=-1)
        num_classes = y_pred.size(-1)
        y_true = F.one_hot(target, num_classes).float()

        if self.average == 'micro':
            tp = (y_true * y_pred).sum()
            fp = ((1 - y_true) * y_pred).sum()
            fn = (y_true * (1 - y_pred)).sum()
            soft_f1 = 2 * tp / (2 * tp + fn + fp + self.eps)
        if self.average == 'macro':
            tp = (y_true * y_pred).sum(dim=0)
            fp = ((1 - y_true) * y_pred).sum(dim=0)
            fn = (y_true * (1 - y_pred)).sum(dim=0)
            soft_f1 = (2 * tp / (2 * tp + fn + fp + self.eps)).mean()

        return 1.0 - soft_f1


class FocalLoss(nn.Module):
    r"""Focal loss introduced in the paper `Focal Loss for Dense Object
    Detection`.

    Args:
        alpha (float): Weighting factor alpha in [0, 1].
        gamma (float): Focusing parameter gamma >= 0.
        is_prob (bool, optional): Wether or not the model's predict is already
            a probability distribution. (Default `False`)
        reduction (str, optional): Specifies the reduction to apply to the
            output: `none` | `mean` | `sum`. Default: `mean`.

            - `none`: no reduction will be applied.
            - `mean`: the sum of the output will be divided by the number
                of elements in the output.
            - `sum`: the output will be summed.

        eps (float): Epsilon value to avoid zero division.
    """
    def __init__(self, alpha=1.0, gamma=1.0, is_prob=False,
                 reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError('reduction should be "none", "mean" or "sum".')
        self.is_prob = is_prob
        self.reduction = reduction
        self.eps = eps

    def forward(self, logit, target):
        r"""
        Args:
            logit (Tensor): Classification model's output before or after
                softmax depend on the `is_prob` argument. Tensor with shape
                (batch_size, num_classes_of_this_task).
            target (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).

        Returns:
            (Tensor): Tensor with shape (batch_size, ) if reduction is `none`.
                Single float tensor else.
        """
        y_prob = (
            logit if self.is_prob
            else F.softmax(logit, dim=-1) + self.eps
        )
        num_classes = y_prob.size(-1)
        y_true = F.one_hot(target, num_classes).float()
        weight = self.alpha * torch.pow(1.0 - y_prob, self.gamma)
        loss = -weight * torch.log(y_prob)
        focal_loss = (y_true * loss).sum(dim=-1)
        if self.reduction == 'none':
            return focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()


class WeightedMultiTaskLoss(nn.Module):
    r"""
    Loss function for multi-task models. It will calculate every single
    task's focal loss and return weighted sum of them.

    Args:
        sample_loss (torch.nn.Module): Loss function implementation in the
            sample level. It should take single task's logit and target as
            input and return a single float value as output. The logit should
            be classification model's output before softmax and with shape
            (batch_size, num_classes_of_this_task). The target should with
            shape (batch_size, ) where each value in this tensor is the class
            index in the range of [0, num_classes_of_this_task).
        task_name2int (Dict[str, int]): Dictionary map task names to ids.
        task_weights (Dict[str, float]): Dictionary with task name as key and
            corresponding task weight as value. The final loss will be the
            weighted sum of all the single task's loss. Only the relative
            magnitude between these values matters cause the final result will
            be normalized.
        return_dict (bool, optional): Wether or not return a dictionary which
            not only contain the weighted sumed loss, but also every single
            task's cross entropy loss.
    """
    def __init__(self, sample_loss, task_name2int,
                 task_weights, return_dict=False):
        super().__init__()
        self.sample_loss = sample_loss
        self.task_name2int = task_name2int
        self.task_weights = task_weights
        self.return_dict = return_dict

    def forward(self, logits, targets, task_type_id):
        r"""
        Args:
            logits (Dict[str, Tensor]): Model's output. Dictionary with task
                name as key and corresponding logits, which has the shape of
                (batch_size, num_classes_of_this_task), as value.
            targets (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).
            task_type_id (Tensor): Tensor with shape (batch_size, ) where each
                value in this tensor is the task index in the range of
                [0, num_tasks).
        """
        losses = {}
        for task_name, logit in logits.items():
            mask = task_type_id == self.task_name2int[task_name]

            # If all the values in the mask are `False`, the sample leval loss
            # will return `nan` or `zero` depending on the sample loss type.
            loss = self.sample_loss(logit[mask], targets[mask])

            losses['loss_' + task_name] = loss

        weighted_sum = 0
        weights = 0
        for loss_name, loss in losses.items():
            if not loss.isnan() and loss > 0:
                weighted_sum += loss * self.task_weights[loss_name[5:]]
                weights += self.task_weights[loss_name[5:]]
        normed_sum = weighted_sum / weights

        if self.return_dict:
            losses['val_loss'] = normed_sum
            return losses
        else:
            return normed_sum


class MultiTaskLossWithUncertainty(nn.Module):
    r"""
    Implement the loss function introduce in the paper: `Multi-Task Learning
    Using Uncertainty to Weight Losses for Scene Geometry and Semantics`.
    Note that this implement only considers classification tasks.

    Args:
        sample_loss (torch.nn.Module): Loss function implementation in the
            sample level. It should take single task's logit and target as
            input and return a single float value as output. The logit should
            be classification model's output before softmax and with shape
            (batch_size, num_classes_of_this_task). The target should with
            shape (batch_size, ) where each value in this tensor is the class
            index in the range of [0, num_classes_of_this_task).
        task_name2int (Dict[str, int]): Dictionary map task names to ids.
        device (torch.device): Which device the model is on.
        return_dict (bool, optional): Wether or not return a dictionary which
            not only contain the final loss, but also every single task's
            uncertainty value sigma^2.
    """
    def __init__(self, sample_loss, task_name2int, device, return_dict=False):
        super().__init__()
        self.task_name2int = task_name2int
        self.sample_loss = sample_loss
        self.return_dict = return_dict

        # Register a trainable parameter which contain the logarithm of the
        # task uncertainty (sigma^2) estimates for each task. The uncertainties
        # are all initialized to 2.0 (sigma^2 = 2.0) which means that this
        # parameter should be initialized to log(2.0).
        self.log_variance = nn.Parameter(
            torch.full(
                size=(len(task_name2int), ),
                fill_value=math.log(2.0),
                device=device
            )
        )

    def forward(self, logits, targets, task_type_id):
        r"""
        Args:
            logits (Dict[str, Tensor]): Model's output. Dictionary with task
                name as key and corresponding logits, which has the shape of
                (batch_size, num_classes_of_this_task), as value.
            targets (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).
            task_type_id (Tensor): Tensor with shape (batch_size, ) where each
                value in this tensor is the task index in the range of
                [0, num_tasks).
        """
        losses = []
        for task_name, logit in logits.items():
            mask = task_type_id == self.task_name2int[task_name]

            # If all the values in the mask are `False`, the sample leval loss
            # will return `nan` or `zero` depending on the sample loss type.
            loss = self.sample_loss(logit[mask], targets[mask])

            losses.append(loss)

        # There may be some `nan` or `zero` values in `losses` due to the
        # absence of a specific type of task sample in this batch, so be
        # careful to handle these values.
        final_loss = 0
        weights = torch.exp(-self.log_variance)
        for i, loss in enumerate(losses):
            weight = weights[i]
            if not loss.isnan():
                final_loss += (
                    loss * weight + torch.log(torch.sqrt(1.0 / weight))
                )

        if self.return_dict:
            ret = {'loss': final_loss}
            for i, task_name in enumerate(self.task_name2int):
                ret['weight_' + task_name] = weights[i]
            return ret
        else:
            return final_loss


class DTPMultiTaskLoss(nn.Module):
    r"""
    Implement the loss function introduce in the paper: `Dynamic Task
    Prioritization for Multitask Learning`.

    Args:
        sample_loss (torch.nn.Module): Loss function implementation in the
            sample level. It should take single task's logit and target as
            input and return a single float value as output. The logit should
            be classification model's output before softmax and with shape
            (batch_size, num_classes_of_this_task). The target should with
            shape (batch_size, ) where each value in this tensor is the class
            index in the range of [0, num_classes_of_this_task).
        sample_kpi (torch.nn.Module): KPI calculation implementation in the
            sample level. It should task the same inputs as `sample_loss` and
            return a single float value between [0, 1] as this task's key
            performance indicator.
        running_alpha (float): The discount factor used when calculate running
            KPI, which is the exponential moving average of KPI. Larger values
            of alpha prioritize more recent examples.
        task_alpha (float): Alpha value used when calculate the weights of each
            task in the focal loss like manner.
        task_gamma (float): Gamma value used when calculate the weights of ecah
            task in the focal loss like manner.
        task_name2int (Dict[str, int]): Dictionary map task names to ids.
        device (torch.device): Which device the model is on.
        return_dict (bool, optional): Wether or not return a dictionary which
            not only contain the final loss, but also every single task's
            running KPI.

    """
    def __init__(self, sample_loss, sample_kpi, running_alpha, task_alpha,
                 task_gamma, task_name2int, device, return_dict=False):
        super().__init__()
        self.sample_loss = sample_loss
        self.sample_kpi = sample_kpi
        self.running_alpha = running_alpha
        self.task_alpha = task_alpha
        self.task_gamma = task_gamma
        self.task_name2int = task_name2int
        self.return_dict = return_dict
        self.running_task_kpis = dict(zip(
            task_name2int,
            torch.full((len(task_name2int), ), 0.5, device=device)
        ))

    def forward(self, logits, targets, task_type_id):
        r"""
        Args:
            logits (Dict[str, Tensor]): Model's output. Dictionary with task
                name as key and corresponding logits, which has the shape of
                (batch_size, num_classes_of_this_task), as value.
            targets (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).
            task_type_id (Tensor): Tensor with shape (batch_size, ) where each
                value in this tensor is the task index in the range of
                [0, num_tasks).
        """
        weighted_sum = 0
        weights = 0
        for task_name, logit in logits.items():
            mask = task_type_id == self.task_name2int[task_name]
            if mask.sum() > 0:
                loss = self.sample_loss(logit[mask], targets[mask])
                kpi = self.sample_kpi(logit[mask], targets[mask])
                running_kpi = (
                    (1.0 - self.running_alpha)
                    * self.running_task_kpis[task_name]
                    + self.running_alpha * kpi
                )
                self.running_task_kpis[task_name] = running_kpi
                weight = (
                    -self.task_alpha
                    * torch.pow((1.0 - running_kpi), self.task_gamma)
                    * torch.log(running_kpi)
                )
                weighted_sum += weight * loss
                weights += weight

        normed_sum = weighted_sum / weights

        if self.return_dict:
            ret = {'loss': normed_sum}
            for task_name, running_kpi in self.running_task_kpis.items():
                ret['running_kpi_' + task_name] = running_kpi
            return ret
        else:
            return normed_sum


class NLPCLoss(nn.Module):
    r"""
    Convenience wrapper module for different sample level loss function and
    task weights adjust methods.

    Args:
        config (Dict): Model configuration dictionary.
        task_name2int (Dict[str, int]): Dictionary map task names to ids.
        split (str): Which split the evaluate data belong to. Should be `train`
            or `val`.
        device (torch.device): Which device the model is on.
    """
    def __init__(self, config, task_name2int, split, device):
        super().__init__()
        if 'sample_loss' not in config or config['sample_loss'] == 'ce':
            sample_loss = nn.CrossEntropyLoss()
        elif config['sample_loss'] == 'focal':
            sample_loss = FocalLoss(
                alpha=config['sample_alpha'],
                gamma=config['sample_gamma']
            )
        elif config['sample_loss'] == 'f1':
            sample_loss = SoftF1Loss(average=config['average'])
        else:
            raise ValueError(
                'sample loss {} not support now.'.format(config['sample_loss'])
            )
        if split not in ['train', 'val']:
            raise ValueError('split should be `train` or `val`.')

        default_task_weights = dict(
            zip(task_name2int.keys(), [1.0] * len(task_name2int))
        )
        if 'task_weights' not in config or config['task_weights'] == 'uni':
            self.task_loss = WeightedMultiTaskLoss(
                sample_loss, task_name2int, default_task_weights,
                return_dict=False if split == 'train' else True
            )
        elif isinstance(config['task_weights'], dict):
            self.task_loss = WeightedMultiTaskLoss(
                sample_loss, task_name2int,
                task_weights=(
                    config['task_weights'] if split == 'train'
                    else default_task_weights
                ),
                return_dict=False if split == 'train' else True
            )
        elif config['task_weights'] == 'uct':
            if split == 'train':
                self.task_loss = MultiTaskLossWithUncertainty(
                    sample_loss, task_name2int, device, return_dict=True
                )
            else:
                self.task_loss = WeightedMultiTaskLoss(
                    sample_loss, task_name2int, default_task_weights,
                    return_dict=True
                )
        elif config['task_weights'] == 'dtp':
            if split == 'train':
                self.task_loss = DTPMultiTaskLoss(
                    sample_loss=sample_loss,
                    sample_kpi=F1ScoreMetric(),
                    running_alpha=config['running_alpha'],
                    task_alpha=config['task_alpha'],
                    task_gamma=config['task_gamma'],
                    task_name2int=task_name2int,
                    device=device,
                    return_dict=True
                )
            else:
                self.task_loss = WeightedMultiTaskLoss(
                    sample_loss, task_name2int, default_task_weights,
                    return_dict=True
                )
        else:
            raise ValueError(
                'task_weights {} not support now.'
                .format(config['task_weights'])
            )

    def forward(self, logits, targets, task_type_id):
        r"""
        Args:
            logits (Dict[str, Tensor]): Model's output. Dictionary with task
                name as key and corresponding logits, which has the shape of
                (batch_size, num_classes_of_this_task), as value.
            targets (Tensor): Target tensor with shape (batch_size, ) where
                each value in this tensor is the class index in the range of
                [0, num_classes_of_this_task).
            task_type_id (Tensor): Tensor with shape (batch_size, ) where each
                value in this tensor is the task index in the range of
                [0, num_tasks).
        """
        return self.task_loss(logits, targets, task_type_id)
