from __future__ import annotations

__all__ = ["BaseLRSchedulerCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from torch.optim import Optimizer

from gravitorch.lr_schedulers.base import LRSchedulerType

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class BaseLRSchedulerCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create a learning rate (LR) scheduler.

    Note that it is not the unique approach to create a LR scheduler.
    Feel free to use other approaches if this approach does not fit your
    needs.
    """

    @abstractmethod
    def create(self, engine: BaseEngine, optimizer: Optimizer | None) -> LRSchedulerType | None:
        r"""Creates an optimizer.

        This method is responsible to register the event handlers
        associated to the LR scheduler. In particular, it should
        register the event to call the ``step`` method of the LR
        scheduler. If the optimizer is ``None``, this function should
        return ``None``  because it does not make sense to define a LR
        scheduler without an optimizer.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            optimizer (``torch.nn.Optimizer``): Specifies the
                optimizer.

        Returns:
            ``LRSchedulerType`` or ``None``: The created LR scheduler
                or ``None`` if there is no LR scheduler to create.
        """
