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

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
        >>> from gravitorch.creators.lr_scheduler import LRSchedulerCreator
        >>> creator = LRSchedulerCreator(
        ...     {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5}
        ... )
        >>> creator
        LRSchedulerCreator(
          (lr_scheduler_config): {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5}
          (lr_scheduler_handler): None
          (add_module_to_engine): True
        )
        >>> engine = create_dummy_engine()
        >>> model = DummyClassificationModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> lr_scheduler = creator.create(engine, optimizer)
        >>> lr_scheduler
        <torch.optim.lr_scheduler.StepLR object at 0x...>
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
        ----
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            optimizer (``torch.nn.Optimizer``): Specifies the
                optimizer.

        Returns:
        -------
            ``LRSchedulerType`` or ``None``: The created LR scheduler
                or ``None`` if there is no LR scheduler to create.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
            >>> from gravitorch.creators.lr_scheduler import LRSchedulerCreator
            >>> creator = LRSchedulerCreator(
            ...     {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5}
            ... )
            >>> engine = create_dummy_engine()
            >>> model = DummyClassificationModel()
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            >>> lr_scheduler = creator.create(engine, optimizer)
            >>> lr_scheduler
            <torch.optim.lr_scheduler.StepLR object at 0x...>
        """
