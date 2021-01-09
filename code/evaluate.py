import torch
import yaml
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from nlpc.dataset import (
    NLPCJointDataset,
    collate_fn_with_padding,
    TASK_NAME2INT,
    OCNLIDataset,
    OCEMOTIONDataset,
    TNEWSDataset
)
from nlpc.model import NLPCModel
from nlpc.loss import NLPCLoss
from nlpc.metric import competition_report
from nlpc.utils import get_args, load_checkpoint


def evaluate(model, val_dataloader, criterion, device, task_name2int):
    r"""
    Evaluate the model on validatioin dataset and return validation loss and
    competition score.

    Args:
        model (torch.nn.Module): The model need to be evaluate.
        val_dataloader (torch.utils.data.DataLoader): Dataloader for validation
            dataset.
        criterion (callable): Loss function implementation.
        device (torch.device): Which device the model is on.

    Returns:
        (tuple): A tuple with validation loss as first element and validation
            score as second element.
    """
    logits = {}
    targets = []
    task_type_id = []
    for batch in tqdm(val_dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            batch_pred = model(batch)

        for k, v in batch_pred.items():
            if k not in logits:
                logits[k] = []
            logits[k].append(v)
        targets.append(batch['target'])
        task_type_id.append(batch['task_type_id'])

    for k, v in logits.items():
        logits[k] = torch.cat(v, dim=0)
    targets = torch.cat(targets, dim=0)
    task_type_id = torch.cat(task_type_id, dim=0)

    val_losses = criterion(logits, targets, task_type_id)
    val_report = competition_report(
        logits, targets, task_type_id, task_name2int
    )

    return val_losses, val_report


if __name__ == '__main__':
    args = get_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    if args.batch_size > 0:
        config['solver']['batch_size'] = args.batch_size

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = (
        torch.device("cuda", args.gpu_ids[0])
        if args.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    if config['model']['share_architecture'] in ['ocnli', 'cmnli']:
        val_dataset = OCNLIDataset(
            config['dataset'],
            split='val',
            overfit=args.overfit,
            tensor_type='np'
        )
        task_name2int = {'ocnli': TASK_NAME2INT['ocnli']}
    elif config['model']['share_architecture'] == 'ocemotion':
        val_dataset = OCEMOTIONDataset(
            config['dataset'],
            split='val',
            overfit=args.overfit,
            tensor_type='np'
        )
        task_name2int = {'ocemotion': TASK_NAME2INT['ocemotion']}
    elif config['model']['share_architecture'] == 'tnews':
        val_dataset = TNEWSDataset(
            config['dataset'],
            split='val',
            overfit=args.overfit,
            tensor_type='np'
        )
        task_name2int = {'tnews': TASK_NAME2INT['tnews']}
    else:
        val_dataset = NLPCJointDataset(
            config['dataset'],
            split='val',
            overfit=args.overfit,
            tensor_type='np'
        )
        task_name2int = {
            'ocnli': TASK_NAME2INT['ocnli'],
            'ocemotion': TASK_NAME2INT['ocemotion'],
            'tnews': TASK_NAME2INT['tnews']
        }

    batch_size = (
        config['solver']['batch_size']
        // config['solver']['accumulation_steps']
     ) * 4
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.cpu_workers,
            collate_fn=collate_fn_with_padding
        )

    model = NLPCModel(config['model']).to(device)
    if -1 not in args.gpu_ids:
        model = nn.DataParallel(model, args.gpu_ids)

    model_state_dict, _ = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    print('Loaded model checkpoint from {}.'.format(args.load_pthpath))

    criterion = NLPCLoss(config['model'], task_name2int, 'val', device)

    model.eval()
    val_losses, val_report = evaluate(
        model, val_dataloader, criterion, device, task_name2int
    )
    for k, v in val_losses.items():
        print('{} = {:.6f}'.format(k, v))
    for k, v in val_report.items():
        print('{} = {:.6f}'.format(k, v))
