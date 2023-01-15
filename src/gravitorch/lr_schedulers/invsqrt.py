__all__ = ["InverseSquareRootLR"]

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def inverse_square_root(step: int) -> float:
    r"""Computes the inverse square root.

    Args:
        step (int): Specifies the current step.

    Returns:
        float: The associated factor for the current step.
    """
    return 1.0 / math.sqrt(1 + step)


class InverseSquareRootLR(LambdaLR):
    r"""Implementation of Inverse Square Root LR scheduler.

    Args:
        optimizer (``torch.optim.Optimizer``): Specifies the optimizer
            associated to the LR scheduler.
        last_epoch (int, optional): Specifies the index of last epoch.
            Default: ``-1``
        verbose (bool, optional): If ``True``, prints a message to
            stdout for each update. Default: ``False``.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.lr_schedulers import InverseSquareRootLR
        >>> optimizer = torch.optim.SGD(torch.nn.Linear(4, 6).parameters(), lr=0.01)
        >>> scheduler = InverseSquareRootLR(optimizer)
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=inverse_square_root,
            last_epoch=last_epoch,
            verbose=verbose,
        )
