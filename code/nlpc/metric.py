import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score


def decode_logits(logits, task_type_id, task_name2int):
    r"""
    Decode model output logits to predict results.

    Args:
        logits (Dict[str, Tensor]): Model's output. Dictionary with task
            name as key and corresponding logits, which has the shape of
            (batch_size, num_classes_of_this_task), as value.
        task_type_id (Tensor): Tensor with shape (batch_size, ) where each
            value in this tensor is the task index in the range of
            [0, num_tasks).
        task_name2int (Dict[str, int]): Dictionary map task names to ids.

    Returns:
        (Tensor): Decoded predict results tensor with shape (batch_size, ).
    """
    y_pred = torch.zeros_like(task_type_id).long()
    for task_name, logit in logits.items():
        mask = task_type_id == task_name2int[task_name]
        y_pred[mask] = torch.argmax(logit, dim=1)[mask]
    return y_pred


def competition_score(logits, targets, task_type_id, task_name2int):
    r"""
    Compute competition score given the model's raw outputs and targets.

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
        task_name2int (Dict[str, int]): Dictionary map task names to ids.

    Returns:
        (float): A single float value which is this competition's primary
            matric.
    """
    y_pred = decode_logits(logits, task_type_id).cpu().numpy()
    y_true = targets.cpu().numpy()
    task_type_id = task_type_id.cpu().numpy()
    ret = []
    for task_name, task_id in task_name2int.items():
        mask = task_type_id == task_id
        ret.append(f1_score(y_true[mask], y_pred[mask], average='macro'))
    return np.mean(ret)


def competition_report(logits, targets, task_type_id, task_name2int):
    r"""
    Report severale metrics of this competition. Not only the competition
    score, every single task's f1-score score also included in this report.

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
        task_name2int (Dict[str, int]): Dictionary map task names to ids.

    Returns:
        (Dict[str, float]): A dictionay with severl keys and corresponding
            metric values. In addition to `competition score`, every single
            task's f1 score could access with key `TaskName_f1`, which
            `TaskName` are `ocnli`, `ocemotion` and `tnews`.
    """
    y_pred = decode_logits(logits, task_type_id, task_name2int).cpu().numpy()
    y_true = targets.cpu().numpy()
    task_type_id = task_type_id.cpu().numpy()
    ret = {}
    f1_scores = []
    for task_name, task_id in task_name2int.items():
        mask = task_type_id == task_id
        ret['f1_' + task_name] = f1_score(
            y_pred[mask], y_true[mask], average='macro'
        )
        f1_scores.append(ret['f1_' + task_name])
    ret['competition_score'] = np.mean(f1_scores)
    return ret


class F1ScoreMetric(nn.Module):
    r"""
    F1 Score for classification task. In the multi-class case this is the
    average of the soft F1 loss of each class with weighting depending on
    the `average` parameter.

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
        num_classes = logit.size(-1)
        y_pred = logit.argmax(dim=-1)
        y_pred = F.one_hot(y_pred, num_classes).float()
        y_true = F.one_hot(target, num_classes).float()

        if self.average == 'micro':
            tp = (y_true * y_pred).sum()
            fp = ((1 - y_true) * y_pred).sum()
            fn = (y_true * (1 - y_pred)).sum()
            f1 = 2 * tp / (2 * tp + fn + fp + self.eps)
        if self.average == 'macro':
            tp = (y_true * y_pred).sum(dim=0)
            fp = ((1 - y_true) * y_pred).sum(dim=0)
            fn = (y_true * (1 - y_pred)).sum(dim=0)
            f1 = (2 * tp / (2 * tp + fn + fp + self.eps)).mean()

        return f1
