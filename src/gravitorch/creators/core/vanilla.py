from __future__ import annotations

__all__ = ["VanillaCoreCreator"]


from torch import nn
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.core.base import BaseCoreCreator
from gravitorch.datasources import setup_datasource
from gravitorch.datasources.base import BaseDataSource
from gravitorch.engines.base import BaseEngine
from gravitorch.lr_schedulers.base import LRSchedulerType, setup_lr_scheduler
from gravitorch.models.utils import setup_model
from gravitorch.optimizers.factory import setup_optimizer
from gravitorch.utils.format import str_indent


class VanillaCoreCreator(BaseCoreCreator):
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
    """

    def __init__(
        self,
        datasource: BaseDataSource | dict,
        model: nn.Module | dict,
        optimizer: Optimizer | dict | None = None,
        lr_scheduler: LRSchedulerType | dict | None = None,
    ) -> None:
        self._datasource = setup_datasource(datasource)
        self._model = setup_model(model)
        self._optimizer = setup_optimizer(model=self._model, optimizer=optimizer)
        self._lr_scheduler = setup_lr_scheduler(
            optimizer=self._optimizer, lr_scheduler=lr_scheduler
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  datasource={str_indent(self._datasource)},\n"
            f"  model={str_indent(self._model)},\n"
            f"  optimizer={str_indent(self._optimizer, num_spaces=4)},\n"
            f"  lr_scheduler={str_indent(self._lr_scheduler)},\n"
            ")"
        )

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
