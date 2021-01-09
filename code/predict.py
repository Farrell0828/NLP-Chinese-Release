import torch
import yaml
import json
import tempfile
import os

from tqdm import tqdm
from zipfile import ZipFile
from torch import nn
from torch.utils.data import DataLoader

from nlpc.dataset import (
    NLPCJointDataset,
    collate_fn_with_padding,
    TASK_NAME2INT,
    OCEMOTION_TARGET2INT,
    TNEWS_TARGET2INT
)
from nlpc.model import NLPCModel
from nlpc.metric import decode_logits
from nlpc.utils import get_args, load_checkpoint


def predict(model, test_dataloader, device, task_name2int):
    r"""
    Evaluate the model on test dataset and return predict results.

    Args:
        model (torch.nn.Module): The model need to be evaluate.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for test
            dataset.
        device (torch.device): Which device the model is on.

    Returns:
        (tuple): A tuple with the follow elements:
            - predict_results: Tensor with shape (num_samples, ).
            - sample_id: Tensor with shape (num_samples, ).
            - task_type_id: Tensor with shape (num_samples, ).
    """
    logits = {}
    task_type_id = []
    sample_id = []
    for batch in tqdm(test_dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            batch_pred = model(batch)

        for k, v in batch_pred.items():
            if k not in logits:
                logits[k] = []
            logits[k].append(v)
        task_type_id.append(batch['task_type_id'])
        sample_id.append(batch['id'])

    for k, v in logits.items():
        logits[k] = torch.cat(v, dim=0)
    task_type_id = torch.cat(task_type_id, dim=0)
    sample_id = torch.cat(sample_id, dim=0)
    predict_results = decode_logits(logits, task_type_id, task_name2int)

    return predict_results, sample_id, task_type_id


def write_to_disk(predict_results, sample_id, task_type_id, output_zippath):
    r"""
    Write model's predict results on testset to a zip file which contain three
    json files.

    Args:
        predict_results (Tensor): Model's predict results. Tensor with shape
            (num_samples, ).
        sample_id (Tensor): The id number of every sample in testset. Tensor
            with shape (num_samples, ).
        task_type_id (Tensor): The task type index every sample in testset.
            Tensor with shape (num_samples, ).

    Returns:
        None.
    """
    int2task_name = dict(
        zip(TASK_NAME2INT.values(), TASK_NAME2INT.keys())
    )
    int2ocemotion_target = dict(
        zip(OCEMOTION_TARGET2INT.values(), OCEMOTION_TARGET2INT.keys())
    )
    int2tnews_target = dict(
        zip(TNEWS_TARGET2INT.values(), TNEWS_TARGET2INT.keys())
    )

    ocnli, ocemotion, tnews = [], [], []
    for i in range(len(predict_results)):
        id = sample_id[i].item()
        label = predict_results[i].item()
        task_name = int2task_name[task_type_id[i].item()]
        if task_name == 'ocnli':
            ocnli.append({'id': str(id), 'label': str(label)})
        elif task_name == 'ocemotion':
            ocemotion.append({
                'id': str(id),
                'label': int2ocemotion_target[label]
            })
        elif task_name == 'tnews':
            tnews.append({
                'id': str(id),
                'label': str(int2tnews_target[label])
            })
        else:
            raise ValueError('Wrong task_name.')

    def write_json(data, file_path):
        with open(file_path, mode='w') as fp:
            for line in data:
                json.dump(line, fp)
                fp.write('\n')

    with tempfile.TemporaryDirectory() as tmpdirname:
        ocnli_path = os.path.join(tmpdirname, 'ocnli_predict.json')
        ocemotion_path = os.path.join(tmpdirname, 'ocemotion_predict.json')
        tnews_path = os.path.join(tmpdirname, 'tnews_predict.json')

        write_json(ocnli, ocnli_path)
        write_json(ocemotion, ocemotion_path)
        write_json(tnews, tnews_path)

        with ZipFile(output_zippath, mode='w') as myzip:
            myzip.write(ocnli_path, arcname='ocnli_predict.json')
            myzip.write(ocemotion_path, arcname='ocemotion_predict.json')
            myzip.write(tnews_path, arcname='tnews_predict.json')


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.save_zippath), exist_ok=True)
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

    test_dataset = NLPCJointDataset(
        config['dataset'],
        split='test',
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
    test_dataloader = DataLoader(
        test_dataset,
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

    model.eval()
    ret = predict(model, test_dataloader, device, task_name2int)
    ret += (args.save_zippath, )
    write_to_disk(*ret)
    print('Writed submission zip file to', args.save_zippath)


if __name__ == '__main__':
    main()
