r"""This module defines the base LR scheduler."""

__all__ = ["BaseLRScheduler", "LRSchedulerType", "setup_lr_scheduler"]

import logging
from typing import Optional, Union

from objectory import AbstractFactory
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


class BaseLRScheduler(_LRScheduler, metaclass=AbstractFactory):
    r"""Defines the base learning rate scheduler."""


# Define this type because there is not a unique class to describe a LR scheduler.
LRSchedulerType = Union[BaseLRScheduler, ReduceLROnPlateau, _LRScheduler]


def setup_lr_scheduler(
    optimizer: Optional[Optimizer],
    lr_scheduler: Union[LRScheduler, dict, None],
) -> Optional[LRScheduler]:
    r"""Sets up a learning rate scheduler.

    Args:
        optimizer (``torch.optim.Optimizer`` or ``None``): Specifies
            the optimizer.
        lr_scheduler (``torch.optim.lr_scheduler._LRScheduler`` or dict
            or ``None``): Specifies the learning rate scheduler or its
            configuration.

    Returns:
        ``torch.optim.lr_scheduler._LRScheduler`` or ``None``: An
            instantiated learning rate scheduler or ``None`` if it is
            not possible to instantiate a learning rate scheduler.
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
