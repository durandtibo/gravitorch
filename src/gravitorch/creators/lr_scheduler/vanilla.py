__all__ = ["VanillaLRSchedulerCreator"]

import logging
from typing import Optional, Union

from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.engines.base import BaseEngine
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import setup_handler
from gravitorch.lr_schedulers.base import LRSchedulerType, setup_lr_scheduler
from gravitorch.utils.format import str_indent

logger = logging.getLogger(__name__)


class VanillaLRSchedulerCreator(BaseLRSchedulerCreator):
    r"""Implements a vanilla a learning rate (LR) scheduler creator.

    This LR scheduler creator has two main inputs: an input to
    configure the LR scheduler and one to manage the LR scheduler.
    The LR scheduler manager is responsible to create events to
    control the LR scheduler.

    Args:
        lr_scheduler_config (dict or ``None``): Specifies the LR
            scheduler configuration. If ``None``, no LR scheduler
            is created and ``None`` will be returned by the ``create``
            method. Default: ``None``
        lr_scheduler_handler (``BaseLRSchedulerManager`` or dict or
            ``None``): Specifies the LR scheduler handler. The LR
            scheduler manager is used only if the LR scheduler can
            be created. If ``None``, no LR scheduler manager is
            created and the user is responsible to manage the LR
            scheduler. Default: ``None``
        add_module_to_engine (bool, optional): If ``True``, the LR
            scheduler is added to the engine state, so the LR
            scheduler state is stored when the engine creates a
            checkpoint. Default: ``True``
    """

    def __init__(
        self,
        lr_scheduler_config: Union[dict, None] = None,
        lr_scheduler_handler: Union[BaseHandler, dict, None] = None,
        add_module_to_engine: bool = True,
    ):
        self._lr_scheduler_config = lr_scheduler_config
        self._lr_scheduler_manager = setup_handler(lr_scheduler_handler)
        logger.info(f"lr_scheduler_handler:\n{lr_scheduler_handler}")
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  lr_scheduler_handler={str_indent(str(self._lr_scheduler_manager))},\n"
            f"  add_module_to_engine={self._add_module_to_engine},\n"
            ")"
        )

    def create(
        self, engine: BaseEngine, optimizer: Optional[Optimizer]
    ) -> Optional[LRSchedulerType]:
        lr_scheduler = setup_lr_scheduler(
            optimizer=optimizer, lr_scheduler=self._lr_scheduler_config
        )
        if lr_scheduler is None:
            return None

        logger.info(f"lr_scheduler:\n{lr_scheduler}")
        if self._add_module_to_engine:
            logger.info(f"Adding a LR scheduler to the engine state (key: {ct.LR_SCHEDULER})...")
            engine.add_module(ct.LR_SCHEDULER, lr_scheduler)

        if self._lr_scheduler_manager:
            logger.info("Attaching a LR scheduler manager to the engine...")
            self._lr_scheduler_manager.attach(engine=engine)
        else:
            logger.warning(
                "No LR scheduler manager is set. If you do not use a LR scheduler manager, you "
                "need to manage 'manually' the LR scheduler"
            )
        return lr_scheduler
