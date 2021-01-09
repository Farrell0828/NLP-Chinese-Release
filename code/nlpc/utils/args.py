import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/config.yml',
        help='Path to the configration yaml file.'
    )
    parser.add_argument(
        '--gpu-ids',
        nargs="+",
        type=int,
        default=0,
        help="List of ids of GPUs to use."
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="Number of CPU workers for dataloader.",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Overfit model on a few examples, meant for debugging."
    )
    parser.add_argument(
        "--save-dirpath",
        default="checkpoints/",
        help="Path of directory to save checkpoints and logs."
    )
    parser.add_argument(
        "--load-pthpath",
        default="",
        help="Path to .pth checkpoint need load when train, evaluate and"
        " predict."
    )
    parser.add_argument(
        "--batch-size",
        default=-1,
        type=int,
        help="Specify the batch size. Will overwrite the corresponding item in"
        " the configuration file if it is a positive number."
    )
    parser.add_argument(
        "--save-zippath",
        default="submissions/submission.zip",
        help="Path of the submission zip file."
    )
    parser.add_argument(
        "--seed",
        default=606,
        type=int,
        help="Random seed."
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Wether or not evaluate on validation set. If True, will use all"
        " data to train."
    )
    return parser.parse_args()
