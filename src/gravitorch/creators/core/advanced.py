from __future__ import annotations

__all__ = ["AdvancedCoreCreator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.creators.core.base import BaseCoreCreator
from gravitorch.creators.datasource.base import (
    BaseDataSourceCreator,
    setup_datasource_creator,
)
from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.creators.lr_scheduler.factory import setup_lr_scheduler_creator
from gravitorch.creators.model.base import BaseModelCreator, setup_model_creator
from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.creators.optimizer.factory import setup_optimizer_creator

if TYPE_CHECKING:
    from gravitorch.datasources import BaseDataSource
    from gravitorch.engines import BaseEngine
    from gravitorch.lr_schedulers import LRSchedulerType


class AdvancedCoreCreator(BaseCoreCreator):
    r"""Implements an advanced core engine moules creator.

    Args:
    ----
        datasource_creator (``BaseDataSourceCreator`` or dict):
            Specifies the datasource creator or its configuration.
        model_creator (``BaseModelCreator`` or dict): Specifies the
            model creator or its configuration.
        optimizer_creator (``BaseOptimizerCreator`` or dict or
            ``None`): Specifies the optimizer creator or its
            configuration. Default: ``None``
        lr_scheduler_creator (``BaseLRSchedulerCreator`` or dict or
            ``None`): Specifies the LR scheduler creator or its
            configuration. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.core import AdvancedCoreCreator
        >>> from gravitorch.creators.datasource import DataSourceCreator
        >>> from gravitorch.creators.model import ModelCreator
        >>> from gravitorch.creators.optimizer import OptimizerCreator
        >>> from gravitorch.creators.lr_scheduler import LRSchedulerCreator
        >>> from gravitorch.testing import create_dummy_engine
        >>> creator = AdvancedCoreCreator(
        ...     datasource_creator=DataSourceCreator(
        ...         {"_target_": "gravitorch.testing.DummyDataSource"}
        ...     ),
        ...     model_creator=ModelCreator(
        ...         {"_target_": "gravitorch.testing.DummyClassificationModel"}
        ...     ),
        ...     optimizer_creator=OptimizerCreator({"_target_": "torch.optim.SGD", "lr": 0.01}),
        ...     lr_scheduler_creator=LRSchedulerCreator(
        ...         {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5}
        ...     ),
        ... )
        >>> creator
        AdvancedCoreCreator(
          (datasource): DataSourceCreator(
              (config): {'_target_': 'gravitorch.testing.DummyDataSource'}
              (attach_to_engine): True
              (add_module_to_engine): True
            )
          (model_creator): ModelCreator(
              (model_config): {'_target_': 'gravitorch.testing.DummyClassificationModel'}
              (attach_model_to_engine): True
              (add_module_to_engine): True
              (device_placement): AutoDevicePlacement(device=cpu)
            )
          (optimizer_creator): OptimizerCreator(add_module_to_engine=True)
          (lr_scheduler_creator): LRSchedulerCreator(
              (lr_scheduler_config): {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5}
              (lr_scheduler_handler): None
              (add_module_to_engine): True
            )
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
        datasource_creator: BaseDataSourceCreator | dict,
        model_creator: BaseModelCreator | dict,
        optimizer_creator: BaseOptimizerCreator | dict | None = None,
        lr_scheduler_creator: BaseLRSchedulerCreator | dict | None = None,
    ) -> None:
        self._datasource_creator = setup_datasource_creator(datasource_creator)
        self._model_creator = setup_model_creator(model_creator)
        self._optimizer_creator = setup_optimizer_creator(optimizer_creator)
        self._lr_scheduler_creator = setup_lr_scheduler_creator(lr_scheduler_creator)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "datasource": self._datasource_creator,
                    "model_creator": self._model_creator,
                    "optimizer_creator": self._optimizer_creator,
                    "lr_scheduler_creator": self._lr_scheduler_creator,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(
        self, engine: BaseEngine
    ) -> tuple[BaseDataSource, Module, Optimizer | None, LRSchedulerType | None]:
        datasource = self._datasource_creator.create(engine=engine)
        model = self._model_creator.create(engine=engine)
        optimizer = self._optimizer_creator.create(engine=engine, model=model)
        lr_scheduler = self._lr_scheduler_creator.create(engine=engine, optimizer=optimizer)
        return datasource, model, optimizer, lr_scheduler
