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
    """

    @abstractmethod
    def create(self, engine: BaseEngine, model: Module) -> Optimizer | None:
        r"""Creates an optimizer.

        This method is responsible to register the event handlers
        associated to the optimizer.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            model (``torch.nn.Module``): Specifies a model.

        Returns:
            ``torch.optim.Optimizer`` or ``None``: The created
                optimizer or ``None`` if no optimizer is created.
        """
