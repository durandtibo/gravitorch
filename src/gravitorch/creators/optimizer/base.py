from __future__ import annotations

__all__ = ["BaseOptimizerCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from torch.nn import Module
from torch.optim import Optimizer

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class BaseOptimizerCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to create an optimizer.

    Note that it is not the unique approach to create an optimizer. Feel
    free to use other approaches if this approach does not fit your
    needs.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
        >>> from gravitorch.creators.optimizer import OptimizerCreator
        >>> creator = OptimizerCreator({"_target_": "torch.optim.SGD", "lr": 0.01})
        >>> creator
        OptimizerCreator(add_module_to_engine=True)
        >>> engine = create_dummy_engine()
        >>> model = DummyClassificationModel()
        >>> optimizer = creator.create(engine, model)
        >>> optimizer
        SGD (
        Parameter Group 0...
            lr: 0.01
            maximize: False
            momentum: 0
            nesterov: False
            weight_decay: 0
        )
    """

    @abstractmethod
    def create(self, engine: BaseEngine, model: Module) -> Optimizer | None:
        r"""Creates an optimizer.

        This method is responsible to register the event handlers
        associated to the optimizer.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            model (``torch.nn.Module``): Specifies a model.

        Returns:
        -------
            ``torch.optim.Optimizer`` or ``None``: The created
                optimizer or ``None`` if no optimizer is created.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
            >>> from gravitorch.creators.optimizer import OptimizerCreator
            >>> creator = OptimizerCreator({"_target_": "torch.optim.SGD", "lr": 0.01})
            >>> engine = create_dummy_engine()
            >>> model = DummyClassificationModel()
            >>> optimizer = creator.create(engine, model)
            >>> optimizer
            SGD (
            Parameter Group 0...
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
        """
