from __future__ import annotations

__all__ = ["OptimizerCreator"]

import logging
from typing import TYPE_CHECKING

from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.optimizers.factory import setup_optimizer

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class OptimizerCreator(BaseOptimizerCreator):
    r"""Implements a vanilla optimizer creator.

    Args:
    ----
        optimizer_config (dict or ``None``, optional): Specifies the
            optimizer configuration. If ``None``, no optimizer is
            created and ``None`` will be returned by the ``create``
            method. Default: ``None``
        add_module_to_engine (bool, optional): If ``True``, the
            optimizer is added to the engine state, so the optimizer
            state is stored when the engine creates a checkpoint.
            Default: ``True``

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

    def __init__(
        self, optimizer_config: dict | None = None, add_module_to_engine: bool = True
    ) -> None:
        self._optimizer_config = optimizer_config
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(add_module_to_engine={self._add_module_to_engine})"

    def create(self, engine: BaseEngine, model: Module) -> Optimizer | None:
        optimizer = setup_optimizer(model=model, optimizer=self._optimizer_config)
        logger.info(f"optimizer:\n{optimizer}")
        if self._add_module_to_engine and optimizer is not None:
            logger.info(f"Adding an optimizer to the engine (key: {ct.OPTIMIZER})...")
            engine.add_module(ct.OPTIMIZER, optimizer)
        return optimizer
