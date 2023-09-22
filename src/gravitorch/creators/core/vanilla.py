from __future__ import annotations

__all__ = ["CoreCreator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from torch import nn
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.core.base import BaseCoreCreator

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource
    from gravitorch.engines import BaseEngine
    from gravitorch.lr_schedulers import LRSchedulerType


class CoreCreator(BaseCoreCreator):
    r"""Implements a simple core engine moules creator.

    This creator does not always "create" the core modules because
    they can already exist. The user is responsible to attach the
    core modules to the engine. This creator only adds the given
    modules to the engine state.

    Args:
    ----
        datasource (``BaseDataSource`` or dict): Specifies the data
            source or its configuration.
        model (``BaseModelCreator`` or dict): Specifies the model
            or its configuration.
        optimizer (``BaseOptimizerCreator`` or dict or ``None`):
            Specifies the optimizer or its configuration.
            Default: ``None``
        lr_scheduler (``BaseLRSchedulerCreator`` or dict or ``None`):
            Specifies the LR scheduler or its configuration.
            Default: ``None``

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

    def __init__(
        self,
        datasource: BaseDataSource | dict,
        model: nn.Module | dict,
        optimizer: Optimizer | dict | None = None,
        lr_scheduler: LRSchedulerType | dict | None = None,
    ) -> None:
        # Local imports to avoid cyclic imports
        from gravitorch.datasources import setup_datasource
        from gravitorch.lr_schedulers import setup_lr_scheduler
        from gravitorch.models import setup_model
        from gravitorch.optimizers import setup_optimizer

        self._datasource = setup_datasource(datasource)
        self._model = setup_model(model)
        self._optimizer = setup_optimizer(model=self._model, optimizer=optimizer)
        self._lr_scheduler = setup_lr_scheduler(
            optimizer=self._optimizer, lr_scheduler=lr_scheduler
        )

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "datasource": self._datasource,
                    "model": self._model,
                    "optimizer": self._optimizer,
                    "lr_scheduler": self._lr_scheduler,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(
        self, engine: BaseEngine
    ) -> tuple[BaseDataSource, nn.Module, Optimizer | None, LRSchedulerType | None]:
        engine.add_module(ct.DATA_SOURCE, self._datasource)
        engine.add_module(ct.MODEL, self._model)
        if self._optimizer:
            engine.add_module(ct.OPTIMIZER, self._optimizer)
        if self._lr_scheduler:
            engine.add_module(ct.LR_SCHEDULER, self._lr_scheduler)
        return self._datasource, self._model, self._optimizer, self._lr_scheduler
