import os
import math
import yaml
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from tensorboardX import SummaryWriter

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
from nlpc.utils import (
    get_args,
    prepare_logger,
    CheckpointManager,
    load_checkpoint
)
from evaluate import evaluate


# For reproducibility.
def seed_everything(my_seed=606):
    np.random.seed(my_seed)
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(config, args, device, logger):
    # Dataset
    if config['model']['share_architecture'] in ['ocnli']:
        train_dataset = OCNLIDataset(
            config['dataset'],
            split='trainval' if args.no_validate else 'train',
            overfit=args.overfit,
            tensor_type='np'
        )
        if not args.no_validate:
            val_dataset = OCNLIDataset(
                config['dataset'],
                split='val',
                overfit=args.overfit,
                tensor_type='np'
            )
        task_name2int = {'ocnli': TASK_NAME2INT['ocnli']}
    elif config['model']['share_architecture'] == 'ocemotion':
        train_dataset = OCEMOTIONDataset(
            config['dataset'],
            split='trainval' if args.no_validate else 'train',
            overfit=args.overfit,
            tensor_type='np'
        )
        if not args.no_validate:
            val_dataset = OCEMOTIONDataset(
                config['dataset'],
                split='val',
                overfit=args.overfit,
                tensor_type='np'
            )
        task_name2int = {'ocemotion': TASK_NAME2INT['ocemotion']}
    elif config['model']['share_architecture'] == 'tnews':
        train_dataset = TNEWSDataset(
            config['dataset'],
            split='trainval' if args.no_validate else 'train',
            overfit=args.overfit,
            tensor_type='np'
        )
        if not args.no_validate:
            val_dataset = TNEWSDataset(
                config['dataset'],
                split='val',
                overfit=args.overfit,
                tensor_type='np'
            )
        task_name2int = {'tnews': TASK_NAME2INT['tnews']}
    else:
        train_dataset = NLPCJointDataset(
            config['dataset'],
            split='trainval' if args.no_validate else 'train',
            overfit=args.overfit,
            tensor_type='np'
        )
        if not args.no_validate:
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

    logger.info(
        'Training set number of samples: {}'.format(len(train_dataset))
    )
    if not args.no_validate:
        logger.info(
            'Validation set number of samples: {}'.format(len(val_dataset))
        )

    assert(
        config['solver']['batch_size']
        % config['solver']['accumulation_steps'] == 0
    )
    actual_batch_size = (
        config['solver']['batch_size']
        // config['solver']['accumulation_steps']
    )
    logger.info('Acture batch size: {}'.format(actual_batch_size))
    logger.info(
        'Gradient accumulation steps: {}'
        .format(config['solver']['accumulation_steps'])
    )
    logger.info(
        'Effective batch size: {}'.format(config['solver']['batch_size'])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=args.cpu_workers,
        collate_fn=collate_fn_with_padding
    )
    if not args.no_validate:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=actual_batch_size * 4,
            shuffle=False,
            num_workers=args.cpu_workers,
            collate_fn=collate_fn_with_padding
        )

    # Model
    model = NLPCModel(config['model']).to(device)
    if -1 not in args.gpu_ids:
        model = nn.DataParallel(model, args.gpu_ids)

    if args.load_pthpath != "":
        model_state_dict, _ = load_checkpoint(args.load_pthpath)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        logger.info(
            'Loaded model checkpoint from {}.'.format(args.load_pthpath)
        )

    # loss
    criterion = NLPCLoss(config['model'], task_name2int, 'train', device)
    if not args.no_validate:
        val_criterion = NLPCLoss(config['model'], task_name2int, 'val', device)

    # Weight decay
    if 'no_decay' in config['solver'].keys():
        no_decay = config['solver']['no_decay']
    else:
        no_decay = []

    transformer_params = [
        item for item in list(model.named_parameters())
        if 'transformer' in item[0]
    ]
    not_transformer_params = [
        item for item in list(model.named_parameters())
        if 'transformer' not in item[0]
    ]

    grouped_parameters = [
        # non-transformer and need decay
        {
            'params': [
                p for n, p in not_transformer_params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': config['solver']['weight_decay'],
            "lr": config['solver']['initial_lr']
        },
        # transformer and need decay
        {
            'params': [
                p for n, p in transformer_params
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': config['solver']['transformer_weight_decay'],
            'lr': (
                config['solver']['transformer_initial_lr']
                if 'transformer_initial_lr' in config['solver']
                else config['solver']['initial_lr']
            )
        },
        # non-transformer and need not decay
        {
            'params': [
                p for n, p in not_transformer_params
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
            'lr': config['solver']['initial_lr']
        },
        # transformer and need not decay
        {
            'params': [
                p for n, p in transformer_params
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
            'lr': (
                config['solver']['transformer_initial_lr']
                if 'transformer_initial_lr' in config['solver']
                else config['solver']['initial_lr']
            )
        }
    ]
    if 'task_weights' in config['model'] \
       and config['model']['task_weights'] == 'uct':
        grouped_parameters.append(
            {
                'params': criterion.parameters(),
                'weight_decay': 0.0,
                'lr': (
                    config['solver']['uct_initial_lr']
                    if 'uct_initial_lr' in config['solver']
                    else config['solver']['initial_lr']
                )
            }
        )

    # Optimizer
    if config['solver']['optimizer'] == 'AdamW':
        optimizer = AdamW(
            grouped_parameters,
            lr=config["solver"]["initial_lr"],
            weight_decay=config['solver']['weight_decay']
        )
    else:
        raise ValueError(
            'optimizer {} not support now.'
            .format(config['solver']['optimizer'])
        )

    # Learning rate schedule
    total_steps = (
        math.ceil(
            len(train_dataloader) / config['solver']['accumulation_steps']
        ) * config['solver']['num_epochs']
        if 'num_epochs' in config['solver']
        else config['solver']['total_steps']
    )
    warmup_steps = (
        math.ceil(total_steps * config['solver']['warmup_fraction'])
        if 'warmup_fraction' in config['solver']
        else config['solver']['warmup_steps']
    )
    validation_steps = (
        config['solver']['validation_steps']
        if 'validation_steps' in config['solver']
        else math.ceil(
            len(train_dataloader) / config['solver']['accumulation_steps']
        )
    ) if not args.no_validate else total_steps

    logger.info('Total steps: {}'.format(total_steps))
    logger.info('Warmup_steps: {}'.format(warmup_steps))
    if not args.no_validate:
        logger.info('Validation steps: {}'.format(validation_steps))

    if config['solver']['lr_schedule'] == 'warmup_linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    elif config['solver']['lr_schedule'] == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    elif config['solver']['lr_schedule'] == 'warmup_cosine_with_hard_restarts':
        num_cycles = config['solver']['num_cycles']
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, num_cycles=num_cycles
        )
    else:
        raise ValueError(
            'Learning rate schedule {} not support not.'
            .format(config['solver']['lr_schedule'])
        )

    # Setup before training
    summary_writer = SummaryWriter(logdir=args.save_dirpath)
    checkpoint_manager = CheckpointManager(
        model, optimizer, args.save_dirpath, overwrite=True, config=config
    )
    accumulation_steps = config['solver']['accumulation_steps']
    forward_steps = 0
    optimizer_steps = 0
    loss = []
    if not args.no_validate:
        best_score = float('-inf')

    # Evaluate before training if loaded pretrained model
    if not args.no_validate and args.load_pthpath != "":
        model.eval()
        val_losses, val_report = evaluate(
            model, val_dataloader, val_criterion, device, task_name2int
        )
        val_score = val_report['competition_score']
        logger.info('Step {} evaluate result:'.format(optimizer_steps))
        for k, v in val_losses.items():
            logger.info('    {} = {:.6f}'.format(k, v))
            if k == 'val_loss':
                summary_writer.add_scalar(
                    "val/loss", v, global_step=optimizer_steps
                )
            else:
                summary_writer.add_scalar(
                    "val/" + k, v, global_step=optimizer_steps
                )
        for k, v in val_report.items():
            logger.info('    {} = {:.6f}'.format(k, v))
            summary_writer.add_scalar(
                "val/" + k, v, global_step=optimizer_steps
            )

    # Training loop
    model.train()
    train_iterator = iter(train_dataloader)
    for _ in range(int(math.ceil(total_steps / validation_steps))):
        for _ in tqdm(range(validation_steps * accumulation_steps)):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                if args.overfit:
                    break
                else:
                    batch = next(train_iterator)
            for key in batch:
                batch[key] = batch[key].to(device)

            batch_output = model(batch)
            batch_loss_output = criterion(
                batch_output, batch['target'], batch['task_type_id']
            )

            if isinstance(batch_loss_output, torch.Tensor):
                batch_loss = batch_loss_output / accumulation_steps
                batch_loss.backward()
                loss.append(batch_loss.detach().cpu().numpy())
            elif (
                isinstance(batch_loss_output, dict)
                and 'task_weights' in config['model']
                and config['model']['task_weights'] in ['uct', 'dtp']
            ):
                batch_loss = batch_loss_output['loss'] / accumulation_steps
                batch_loss.backward()
                loss.append(batch_loss.detach().cpu().numpy())
            else:
                raise ValueError()

            forward_steps += 1

            if forward_steps % accumulation_steps == 0:
                optimizer_steps += 1

                loss = np.sum(loss)
                summary_writer.add_scalar(
                    "train/loss", loss, global_step=optimizer_steps
                )
                loss = []

                if isinstance(batch_loss_output, dict) \
                   and 'task_weights' in config['model'] \
                   and config['model']['task_weights'] == 'uct':
                    for task_name in task_name2int:
                        summary_writer.add_scalar(
                            "train/weight_" + task_name,
                            batch_loss_output["weight_" + task_name],
                            global_step=optimizer_steps
                        )

                if isinstance(batch_loss_output, dict) \
                   and 'task_weights' in config['model'] \
                   and config['model']['task_weights'] == 'dtp':
                    for task_name in task_name2int:
                        summary_writer.add_scalar(
                            "train/running_kpi_" + task_name,
                            batch_loss_output["running_kpi_" + task_name],
                            global_step=optimizer_steps
                        )

                summary_writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"],
                    global_step=optimizer_steps
                )
                summary_writer.add_scalar(
                    "train/transformer_lr", optimizer.param_groups[1]["lr"],
                    global_step=optimizer_steps
                )
                if 'task_weights' in config['model'] \
                   and config['model']['task_weights'] == 'uct':
                    summary_writer.add_scalar(
                        "train/uct_lr", optimizer.param_groups[-1]["lr"],
                        global_step=optimizer_steps
                    )

                if config['solver']['max_grad_norm'] > 0:
                    clip_grad_norm_(
                        model.parameters(),
                        config['solver']['max_grad_norm']
                    )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                torch.cuda.empty_cache()

                if optimizer_steps >= total_steps:
                    break

        # Evaluate on validation set.
        if not args.no_validate:
            model.eval()
            val_losses, val_report = evaluate(
                model, val_dataloader, val_criterion, device, task_name2int
            )
            val_score = val_report['competition_score']
            logger.info('Step {} evaluate result:'.format(optimizer_steps))
            for k, v in val_losses.items():
                logger.info('    {} = {:.6f}'.format(k, v))
                if k == 'val_loss':
                    summary_writer.add_scalar(
                        "val/loss", v, global_step=optimizer_steps
                    )
                else:
                    summary_writer.add_scalar(
                        "val/" + k, v, global_step=optimizer_steps
                    )
            for k, v in val_report.items():
                logger.info('    {} = {:.6f}'.format(k, v))
                summary_writer.add_scalar(
                    "val/" + k, v, global_step=optimizer_steps
                )

            if val_score > best_score:
                checkpoint_manager.step()
                logger.info(
                    '    Validation best score update from {:.6f} to {:.6f}. '
                    'Saved checkpoint to {}'.format(
                        best_score, val_score, args.save_dirpath + 'checkpoint.pth'
                    )
                )
                best_score = val_score
            else:
                logger.info(
                    '    Validation best score not updated since {:.6f}. '
                    'No checkpoint saved.'.format(best_score)
                )
            model.train()
            torch.cuda.empty_cache()
            summary_writer.flush()

    # Save the final model if no validate
    if args.no_validate:
        checkpoint_manager.step()
        logger.info(
            'Saved final checkpoint to {}'.format(
                args.save_dirpath + 'checkpoint.pth'
            )
        )

    summary_writer.close()


if __name__ == '__main__':
    args = get_args()
    seed_everything(my_seed=args.seed)
    os.makedirs(args.save_dirpath, exist_ok=True)
    logger = prepare_logger(args.save_dirpath)
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    logger.info(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        logger.info("{:<20}: {}".format(arg, getattr(args, arg)))

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = (
        torch.device("cuda", args.gpu_ids[0])
        if args.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    train(config, args, device, logger)
