__all__ = ["VanillaOptimizerCreator"]

import logging
from typing import Optional

from torch.nn import Module
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.optimizer.base import BaseOptimizerCreator
from gravitorch.engines.base import BaseEngine
from gravitorch.optimizers import setup_optimizer

logger = logging.getLogger(__name__)


class VanillaOptimizerCreator(BaseOptimizerCreator):
    r"""Implements a vanilla optimizer creator.

    Args:
        optimizer_config (dict or ``None``, optional): Specifies the
            optimizer configuration. If ``None``, no optimizer is
            created and ``None`` will be returned by the ``create``
            method. Default: ``None``
        add_module_to_engine (bool, optional): If ``True``, the
            optimizer is added to the engine state, so the optimizer
            state is stored when the engine creates a checkpoint.
            Default: ``True``
    """

    def __init__(self, optimizer_config: Optional[dict] = None, add_module_to_engine: bool = True):
        self._optimizer_config = optimizer_config
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(add_module_to_engine={self._add_module_to_engine})"

    def create(self, engine: BaseEngine, model: Module) -> Optional[Optimizer]:
        r"""Creates an optimizer.

        This method is responsible to register the event handlers
        associated to the optimizer.

        Args:
            engine (``gravitorch.engines.BaseEngine``): Specifies an
                engine.
            model (``torch.nn.Module``): Specifies a model.

        Returns:
            ``torch.optim.Optimizer`` or ``None``: The created
                optimizer or ``None`` if there is no optimizer to
                create.
        """
        optimizer = setup_optimizer(model=model, optimizer=self._optimizer_config)
        logger.info(f"optimizer:\n{optimizer}")
        if self._add_module_to_engine and optimizer is not None:
            logger.info(f"Adding an optimizer to the engine (key: {ct.OPTIMIZER})...")
            engine.add_module(ct.OPTIMIZER, optimizer)
        return optimizer
