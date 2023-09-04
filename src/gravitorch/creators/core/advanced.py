from __future__ import annotations

__all__ = ["AdvancedCoreCreator"]

from typing import TYPE_CHECKING

from coola.utils import str_indent
from torch.nn import Module
from torch.optim import Optimizer

from gravitorch.creators.core.base import BaseCoreCreator
from gravitorch.creators.datasource.base import (
    BaseDataSourceCreator,
    setup_datasource_creator,
)
from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.creators.lr_scheduler.utils import setup_lr_scheduler_creator
from gravitorch.creators.model.base import BaseModelCreator, setup_model_creator
from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.creators.optimizer.utils import setup_optimizer_creator
from gravitorch.datasources.base import BaseDataSource
from gravitorch.lr_schedulers.base import LRSchedulerType

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


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
        >>> from gravitorch.creators.datasource import VanillaDataSourceCreator
        >>> from gravitorch.creators.model import VanillaModelCreator
        >>> from gravitorch.creators.optimizer import VanillaOptimizerCreator
        >>> from gravitorch.creators.lr_scheduler import VanillaLRSchedulerCreator
        >>> from gravitorch.testing import create_dummy_engine
        >>> creator = AdvancedCoreCreator(
        ...     datasource_creator=VanillaDataSourceCreator(
        ...         {"_target_": "gravitorch.testing.DummyDataSource"}
        ...     ),
        ...     model_creator=VanillaModelCreator(
        ...         {"_target_": "gravitorch.testing.DummyClassificationModel"}
        ...     ),
        ...     optimizer_creator=VanillaOptimizerCreator(
        ...         {"_target_": "torch.optim.SGD", "lr": 0.01}
        ...     ),
        ...     lr_scheduler_creator=VanillaLRSchedulerCreator(
        ...         {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5}
        ...     ),
        ... )
        >>> creator  # doctest: +ELLIPSIS
        AdvancedCoreCreator(
          datasource_creator=VanillaDataSourceCreator(attach_to_engine=True, add_module_to_engine=True),
          model_creator=VanillaModelCreator(
            attach_model_to_engine=True,
            add_module_to_engine=True,
            device_placement=AutoDevicePlacement(device=cpu),
          ),
          optimizer_creator=VanillaOptimizerCreator(add_module_to_engine=True),
          lr_scheduler_creator=VanillaLRSchedulerCreator(
            lr_scheduler_handler=None,
            add_module_to_engine=True,
          ),
        )
        >>> engine = create_dummy_engine()
        >>> datasource, model, optimizer, lr_scheduler = creator.create(engine)
        >>> datasource
        DummyDataSource(
          datasets:
            (train): DummyDataset(num_examples=4, feature_size=4)
            (eval): DummyDataset(num_examples=4, feature_size=4)
          dataloader_creators:
            (train): DataLoaderCreator(
                batch_size : 1
                seed       : 0
                shuffle    : False
              )
            (eval): DataLoaderCreator(
                batch_size : 1
                seed       : 0
                shuffle    : False
              )
        )
        >>> model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
        >>> optimizer  # doctest: +ELLIPSIS
        SGD (
          Parameter Group 0...
              lr: 0.01
              maximize: False
              momentum: 0
              nesterov: False
              weight_decay: 0
        )
        >>> lr_scheduler  # doctest: +ELLIPSIS
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
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  datasource_creator={str_indent(self._datasource_creator)},\n"
            f"  model_creator={str_indent(self._model_creator)},\n"
            f"  optimizer_creator={str_indent(self._optimizer_creator)},\n"
            f"  lr_scheduler_creator={str_indent(self._lr_scheduler_creator)},\n"
            ")"
        )

    def create(
        self, engine: BaseEngine
    ) -> tuple[BaseDataSource, Module, Optimizer | None, LRSchedulerType | None]:
        datasource = self._datasource_creator.create(engine=engine)
        model = self._model_creator.create(engine=engine)
        optimizer = self._optimizer_creator.create(engine=engine, model=model)
        lr_scheduler = self._lr_scheduler_creator.create(engine=engine, optimizer=optimizer)
        return datasource, model, optimizer, lr_scheduler
