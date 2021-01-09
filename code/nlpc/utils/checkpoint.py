"""
A checkpoint manager periodically saves model and optimizer as .pth
files during training.

Checkpoint managers help with experiment reproducibility, they record
the commit SHA of your current codebase in the checkpoint saving
directory. While loading any checkpoint from other commit, they raise a
friendly warning, a signal to inspect commit diffs for potential bugs.
Moreover, they copy experiment hyper-parameters as a YAML config in
this directory.

That said, always run your experiments after committing your changes,
this doesn't account for untracked or staged, but uncommitted changes.
"""
from pathlib import Path

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):
    r"""
    A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Args:
        model (nn.Module): Wrapped model, which needs to be checkpointed.
        optimizer (optim.Optimizer): Wrapped optimizer which needs to be
            checkpointed.
        checkpoint_dirpath (str): Path to an empty or non-existent directory
            to save checkpoints.
        step_size (int, optional): Period of saving checkpoints. (default 1)
        last_epoch (int, optional): The index of last epoch. (default -1)
        overwrite (bool, optional): Wether or not overwrite on the old
            checkpoint when save new checkpoint. Set to `True` when you only
            want keep one checkpoint (Usually the best performance one, but
            this module does not guarantee this. You need to implement it
            outside.) to reduce disk space consumption. (default `False`)

    Example
    --------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt")
    >>> for epoch in range(20):
    ...     for batch in dataloader:
    ...         do_iteration(batch)
    ...     ckpt_manager.step()
    """

    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dirpath,
        step_size=1,
        last_epoch=-1,
        overwrite=False,
        **kwargs,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.overwrite = overwrite
        self.init_directory(**kwargs)

    def init_directory(self, config={}):
        r"""
        Initialize empty checkpoint directory and save hyper-parameters config
        in this directory to associate checkpoints with their hyper-parameters.
        """

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            config,
            open(str(self.ckpt_dirpath / "config.yml"), "w"),
            default_flow_style=False,
        )

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet. """

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if not self.last_epoch % self.step_size:
            file_name = (
                f"checkpoint_{self.last_epoch}.pth"
                if not self.overwrite
                else "checkpoint.pth"
            )
            torch.save(
                {
                    "model": self._model_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_dirpath / file_name,
            )

    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath, map_device=None):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it.

    Args:
        checkpoint_pthpath (str or pathlib.Path): Path to saved checkpoint
            (as created by ``CheckpointManager``).
        map_device (None or torch.device): Map the loaded model state
            dictionary to the device specified if it is not None.
            (Default None).

    Returns:
        nn.Module, optim.Optimizer
            Model and optimizer state dicts loaded from checkpoint.
    """

    components = torch.load(checkpoint_pthpath, map_location=map_device)
    return components["model"], components["optimizer"]
