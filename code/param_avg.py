import argparse
import torch

from collections import OrderedDict
from nlpc.utils import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-pthpaths',
        nargs='+',
        help='Paths ot checkpoints.'
    )
    parser.add_argument(
        '--output-pthpath',
        help='Path to merged and averaged checkpoints.'
    )
    return parser.parse_args()


def load_checkpoints(paths):
    r"""
    Load checkpoints from paths.

    Args:
        paths (List[str]): List of paths.

    Returns:
        tuple[collections.OrderedDict]: Loaded checkpoints.
    """
    map_device = torch.device('cpu')
    ret = []
    for path in paths:
        model_state_dict, _ = load_checkpoint(path, map_device=map_device)
        ret.append(model_state_dict, )
        print('Loaded checkpoint', path)
    return ret


def checkpoints_average(model_state_dicts):
    r"""
    This function take several model state dictionarys as input and merged them
    together. If there is same parameter name in different model state
    dictionarys, the corresponding parameters should have the same shape and
    will be averaged.

    Args:
        model_state_dicts (List[collections.OrderedDict]): List of model state
            dictionarys.

    Returns:
        (collections.OrderedDict): Merged and averaged mdoel state dictionary.
    """
    new_state_dict = OrderedDict()
    count_dict = {}
    for model_state_dict in model_state_dicts:
        for name, param in model_state_dict.items():
            if name in new_state_dict and 'position_ids' not in name:
                new_state_dict[name] += param
                count_dict[name] += 1
            else:
                new_state_dict[name] = param
                count_dict[name] = 1
    for name, param in new_state_dict.items():
        print('param:', name)
        if count_dict[name] > 1:
            new_state_dict[name] /= count_dict[name]
    return new_state_dict


def save_checkpoint(model_state_dict, output_pthpath):
    r"""
    Save model state dictionary. It will also create and save a dummy optimizer
    to keep compatible with `load_checkpoint` used in training or evaluation.
    """
    torch.save(
        {
            "model": model_state_dict,
            "optimizer": None,
        },
        output_pthpath
    )
    print('Saved new checkpoint to', output_pthpath)


def main():
    args = get_args()
    checkpoints = load_checkpoints(args.input_pthpaths)
    new_checkpoint = checkpoints_average(checkpoints)
    save_checkpoint(new_checkpoint, args.output_pthpath)


if __name__ == '__main__':
    main()
