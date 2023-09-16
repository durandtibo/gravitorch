r"""This module defines the base LR scheduler."""
from __future__ import annotations

__all__ = ["BaseLRScheduler", "LRSchedulerType", "setup_lr_scheduler"]

import logging
from typing import Union

from objectory import AbstractFactory
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseLRScheduler(LRScheduler, metaclass=AbstractFactory):
    r"""Defines the base learning rate scheduler."""


# Define this type because there is not a unique class to describe a LR scheduler.
LRSchedulerType = Union[BaseLRScheduler, ReduceLROnPlateau, LRScheduler]


def setup_lr_scheduler(
    optimizer: Optimizer | None,
    lr_scheduler: LRSchedulerType | dict | None,
) -> LRSchedulerType | None:
    r"""Sets up a learning rate scheduler.

    Args:
    ----
        optimizer (``torch.optim.Optimizer`` or ``None``): Specifies
            the optimizer.
        lr_scheduler (``torch.optim.lr_scheduler._LRScheduler`` or dict
            or ``None``): Specifies the learning rate scheduler or its
            configuration.

    Returns:
    -------
        ``torch.optim.lr_scheduler.LRScheduler`` or ``None``: An
            instantiated learning rate scheduler or ``None`` if it is
            not possible to instantiate a learning rate scheduler.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.lr_schedulers import setup_lr_scheduler
        >>> optimizer = torch.optim.SGD(torch.nn.Linear(4, 6).parameters(), lr=0.01)
        >>> lr_scheduler = setup_lr_scheduler(
        ...     optimizer=optimizer,
        ...     lr_scheduler={"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ... )
        >>> lr_scheduler
        <torch.optim.lr_scheduler.StepLR object at 0x...>
    """
    if lr_scheduler is None:
        logger.info("No LR scheduler")
        return None
    if optimizer is None:
        logger.info("The LR scheduler is not initialized because there is no optimizer")
        return None
    if isinstance(lr_scheduler, dict):
        logger.info(
            "Initializing a LR scheduler from its configuration... "
            f"{str_target_object(lr_scheduler)}"
        )
        lr_scheduler = BaseLRScheduler.factory(optimizer=optimizer, **lr_scheduler)
    if hasattr(lr_scheduler, "optimizer") and lr_scheduler.optimizer != optimizer:
        logger.warning(
            "The optimizer associated to the LR scheduler is different from the optimizer. "
            "Please verify that the LR scheduler was correctly initialized with the right "
            "optimizer."
        )
    return lr_scheduler
