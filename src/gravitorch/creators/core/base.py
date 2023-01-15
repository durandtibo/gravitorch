from __future__ import annotations

__all__ = ["BaseCoreCreator", "setup_core_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.datasources.base import BaseDataSource
from gravitorch.lr_schedulers.base import LRSchedulerType
from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseCoreCreator(ABC, metaclass=AbstractFactory):
    """Defines the base class to create some core engine modules.

    In MLTorch, the core engine modules are:

        - data source
        - model
        - optimizer
        - LR scheduler

    Note it is possible to create these core modules without using
    this class.
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine
    ) -> tuple[BaseDataSource, Module, Optimizer | None, LRSchedulerType | None]:
        r"""Creates the core engine modules.

        This method is responsible to register the event handlers
        associated to the core engine modules.

        Args:
            engine (``BaseEngine``): Specifies the engine.

        Returns:
            tuple with 4 values with the following structure:
                - ``gravitorch.datasources.BaseDataSource``: The
                    initialized data source.
                - ``torch.nn.Module``: The instantiated model.
                - ``torch.optim.Optimizer`` or ``None``: The
                    instantiated optimizer or ``None`` if there
                    is no optimizer (evaluation mode only).
                - ``LRSchedulerType`` or ``None``: The instantiated
                    learning rate (LR) scheduler or ``None`` if
                    there is no learning rate scheduler.
        """


def setup_core_creator(creator: BaseCoreCreator | dict) -> BaseCoreCreator:
    r"""Sets up the core engine modules creator.

    The core engine modules creator is instantiated from its
    configuration by using the ``BaseCoreModulesCreator`` factory
    function.

    Args:
        creator (``BaseCoreCreator`` or dict): Specifies the
            core engine modules creator or its configuration.

    Returns:
        ``BaseCoreCreator``: The instantiated core engine
            modules creator.
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a core engine modules creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseCoreCreator.factory(**creator)
    return creator
