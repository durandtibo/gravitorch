from __future__ import annotations

__all__ = ["BaseCoreCreator", "is_core_creator_config", "setup_core_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource
    from gravitorch.engines import BaseEngine
    from gravitorch.lr_schedulers import LRSchedulerType

logger = logging.getLogger(__name__)


class BaseCoreCreator(ABC, metaclass=AbstractFactory):
    """Defines the base class to create some core engine modules.

    In MLTorch, the core engine modules are:

        - datasource
        - model
        - optimizer
        - LR scheduler

    Note it is possible to create these core modules without using
    this class.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> from gravitorch.creators.core import CoreCreator
        >>> creator = CoreCreator(
        ...     datasource={"_target_": "gravitorch.testing.DummyDataSource"},
        ...     model={"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...     optimizer={"_target_": "torch.optim.SGD", "lr": 0.01},
        ...     lr_scheduler={"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ... )
        >>> creator
        CoreCreator(
          (datasource): DummyDataSource(
              (datasets):
                (train): DummyDataset(num_examples=4, feature_size=4)
                (eval): DummyDataset(num_examples=4, feature_size=4)
              (dataloader_creators):
                (train): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
                (eval): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
            )
          (model): DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
          (optimizer): SGD (
            Parameter Group 0...
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
          (lr_scheduler): <torch.optim.lr_scheduler.StepLR object at 0x...>
        )
        >>> engine = create_dummy_engine()
        >>> datasource, model, optimizer, lr_scheduler = creator.create(engine)
        >>> datasource
        DummyDataSource(
          (datasets):
            (train): DummyDataset(num_examples=4, feature_size=4)
            (eval): DummyDataset(num_examples=4, feature_size=4)
          (dataloader_creators):
            (train): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
            (eval): DataLoaderCreator(
                (seed): 0
                (batch_size): 1
                (shuffle): False
              )
        )
        >>> model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
        >>> optimizer
        SGD (
          Parameter Group 0...
              lr: 0.01
              maximize: False
              momentum: 0
              nesterov: False
              weight_decay: 0
        )
        >>> lr_scheduler
        <torch.optim.lr_scheduler.StepLR object at 0x...>
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine
    ) -> tuple[BaseDataSource, Module, Optimizer | None, LRSchedulerType | None]:
        r"""Creates the core engine modules.

        This method is responsible to register the event handlers
        associated to the core engine modules.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Returns:
        -------
            tuple with 4 values with the following structure:
                - ``gravitorch.datasources.BaseDataSource``: The
                    initialized datasource.
                - ``torch.nn.Module``: The instantiated model.
                - ``torch.optim.Optimizer`` or ``None``: The
                    instantiated optimizer or ``None`` if there
                    is no optimizer (evaluation mode only).
                - ``LRSchedulerType`` or ``None``: The instantiated
                    learning rate (LR) scheduler or ``None`` if
                    there is no learning rate scheduler.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.testing import create_dummy_engine
            >>> from gravitorch.creators.core import CoreCreator
            >>> creator = CoreCreator(
            ...     datasource={"_target_": "gravitorch.testing.DummyDataSource"},
            ...     model={"_target_": "gravitorch.testing.DummyClassificationModel"},
            ...     optimizer={"_target_": "torch.optim.SGD", "lr": 0.01},
            ...     lr_scheduler={"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5},
            ... )
            >>> engine = create_dummy_engine()
            >>> datasource, model, optimizer, lr_scheduler = creator.create(engine)
            >>> datasource
            DummyDataSource(
              (datasets):
                (train): DummyDataset(num_examples=4, feature_size=4)
                (eval): DummyDataset(num_examples=4, feature_size=4)
              (dataloader_creators):
                (train): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
                (eval): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
            )
            >>> model
            DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
            >>> optimizer
            SGD (
              Parameter Group 0...
                  lr: 0.01
                  maximize: False
                  momentum: 0
                  nesterov: False
                  weight_decay: 0
            )
            >>> lr_scheduler
            <torch.optim.lr_scheduler.StepLR object at 0x...>
        """


def is_core_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseCoreCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseCoreCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.core import is_core_creator_config
        >>> is_core_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.core.CoreCreator",
        ...         "datasource": {"_target_": "gravitorch.testing.DummyDataSource"},
        ...         "model": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...         "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        ...         "lr_scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseCoreCreator)


def setup_core_creator(creator: BaseCoreCreator | dict) -> BaseCoreCreator:
    r"""Sets up the core engine modules creator.

    The core engine modules creator is instantiated from its
    configuration by using the ``BaseCoreModulesCreator`` factory
    function.

    Args:
    ----
        creator (``BaseCoreCreator`` or dict): Specifies the
            core engine modules creator or its configuration.

    Returns:
    -------
        ``BaseCoreCreator``: The instantiated core engine
            modules creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.core import setup_core_creator
        >>> creator = setup_core_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.core.CoreCreator",
        ...         "datasource": {"_target_": "gravitorch.testing.DummyDataSource"},
        ...         "model": {"_target_": "gravitorch.testing.DummyClassificationModel"},
        ...         "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
        ...         "lr_scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ...     }
        ... )
        >>> creator
        CoreCreator(
          (datasource): DummyDataSource(
              (datasets):
                (train): DummyDataset(num_examples=4, feature_size=4)
                (eval): DummyDataset(num_examples=4, feature_size=4)
              (dataloader_creators):
                (train): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
                (eval): DataLoaderCreator(
                    (seed): 0
                    (batch_size): 1
                    (shuffle): False
                  )
            )
          (model): DummyClassificationModel(
              (linear): Linear(in_features=4, out_features=3, bias=True)
              (criterion): CrossEntropyLoss()
            )
          (optimizer): SGD (
            Parameter Group 0...
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
          (lr_scheduler): <torch.optim.lr_scheduler.StepLR object at 0x...>
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a core engine modules creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseCoreCreator.factory(**creator)
    if not isinstance(creator, BaseCoreCreator):
        logger.warning(f"creator is not a `BaseCoreCreator` (received: {type(creator)})")
    return creator
