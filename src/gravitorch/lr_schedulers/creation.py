__all__ = [
    "create_linear_warmup_cosine_decay_lr",
    "create_linear_warmup_linear_decay_lr",
    "create_sequential_lr",
]

from collections.abc import Sequence
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from gravitorch.lr_schedulers.base import LRSchedulerType, setup_lr_scheduler


def create_sequential_lr(
    optimizer: Optimizer,
    schedulers: Sequence[Union[LRSchedulerType, dict]],
    milestones: list[int],
    last_epoch: int = -1,
) -> SequentialLR:
    r"""Instantiates a sequential LR scheduler from its configuration.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            associated to the LR scheduler.
        schedulers (sequence): Specifies the sequence of chained
            schedulers or their configurations.
        milestones (list): Specifies the list of integers that
            reflects milestone points.
        last_epoch (int, optional): Specifies the index of last epoch.
            Default: ``-1``

    Returns:
        ``SequentialLR``: The instantiated sequential LR scheduler.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.lr_schedulers import create_sequential_lr
        >>> optimizer = torch.optim.SGD(torch.nn.Linear(4, 6).parameters(), lr=0.01)
        >>> scheduler = create_sequential_lr(
        ...     optimizer,
        ...     schedulers=[
        ...         {
        ...             "_target_": "torch.optim.lr_scheduler.LinearLR",
        ...             "start_factor": 0.001,
        ...             "end_factor": 1.0,
        ...             "total_iters": 2,
        ...         },
        ...         {
        ...             "_target_": "torch.optim.lr_scheduler.LinearLR",
        ...             "start_factor": 1.0,
        ...             "end_factor": 0.001,
        ...             "total_iters": 8,
        ...         },
        ...     ],
        ...     milestones=[2],
        ... )
        <torch.optim.lr_scheduler.SequentialLR at 0x012345678>
    """
    return SequentialLR(
        optimizer,
        schedulers=[setup_lr_scheduler(optimizer, scheduler) for scheduler in schedulers],
        milestones=milestones,
        last_epoch=last_epoch,
    )


def create_linear_warmup_cosine_decay_lr(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    start_factor: float = 0.001,
    end_lr: float = 1e-6,
) -> SequentialLR:
    r"""Creates a LR scheduler with linear warm-up and cosine decay.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            associated to the LR scheduler.
        num_warmup_steps (int): Specifies the number of steps for the
            warmup phase.
        num_total_steps (int): Specifies the number of total steps.
        start_factor (float, optional): Specifies the multiplication
            factor for the first step (``0``). Default: ``0.001``
        end_lr (float, optional): Specifies the learning rate for the
            last step (``num_total_steps``). Default: ``1e-6``

    Returns:
        ``torch.optim.lr_scheduler.SequentialLR``: The instantiated
            learning rate scheduler.
    """
    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=num_warmup_steps
    )
    decay = CosineAnnealingLR(optimizer, T_max=num_total_steps - num_warmup_steps, eta_min=end_lr)
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[num_warmup_steps])


def create_linear_warmup_linear_decay_lr(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
    start_factor: float = 0.001,
    end_factor: float = 0.001,
) -> SequentialLR:
    r"""Creates a LR scheduler with linear warm-up and linear decay.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            associated to the LR scheduler.
        num_warmup_steps (int): Specifies the number of steps for the
            warmup phase.
        num_total_steps (int): Specifies the number of total steps.
        start_factor (float, optional): Specifies the multiplication
            factor for the first step (``0``). Default: ``0.001``
        end_factor (float, optional): Specifies the multiplication
            factor for the last step (``num_total_steps``).
            Default: ``0.001``

    Returns:
        ``torch.optim.lr_scheduler.SequentialLR``: The instantiated
            learning rate scheduler.
    """
    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=num_warmup_steps
    )
    decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=num_total_steps - num_warmup_steps,
    )
    return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[num_warmup_steps])
