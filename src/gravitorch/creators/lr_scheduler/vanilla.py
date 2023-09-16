from __future__ import annotations

__all__ = ["LRSchedulerCreator"]

import logging
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from torch.optim import Optimizer

from gravitorch import constants as ct
from gravitorch.creators.lr_scheduler.base import BaseLRSchedulerCreator
from gravitorch.lr_schedulers.base import LRSchedulerType, setup_lr_scheduler
from gravitorch.utils.format import str_pretty_json

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine
    from gravitorch.handlers import BaseHandler

logger = logging.getLogger(__name__)


class LRSchedulerCreator(BaseLRSchedulerCreator):
    r"""Implements a vanilla a learning rate (LR) scheduler creator.

    This LR scheduler creator has two main inputs: an input to
    configure the LR scheduler and one to manage the LR scheduler.
    The LR scheduler manager is responsible to create events to
    control the LR scheduler.

    Args:
    ----
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

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.testing import create_dummy_engine, DummyClassificationModel
        >>> from gravitorch.creators.lr_scheduler import LRSchedulerCreator
        >>> creator = LRSchedulerCreator(
        ...     {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 5}
        ... )
        >>> creator
        LRSchedulerCreator(
          (lr_scheduler_config): {'_target_': 'torch.optim.lr_scheduler.StepLR', 'step_size': 5}
          (lr_scheduler_handler): None
          (add_module_to_engine): True
        )
        >>> engine = create_dummy_engine()
        >>> model = DummyClassificationModel()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> lr_scheduler = creator.create(engine, optimizer)
        >>> lr_scheduler
        <torch.optim.lr_scheduler.StepLR object at 0x...>
    """

    def __init__(
        self,
        lr_scheduler_config: dict | None = None,
        lr_scheduler_handler: BaseHandler | dict | None = None,
        add_module_to_engine: bool = True,
    ) -> None:
        self._lr_scheduler_config = lr_scheduler_config

        # Local import to avoid cyclic import
        from gravitorch.handlers.utils import setup_handler

        self._lr_scheduler_handler = setup_handler(lr_scheduler_handler)
        logger.info(f"lr_scheduler_handler:\n{lr_scheduler_handler}")
        self._add_module_to_engine = bool(add_module_to_engine)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "lr_scheduler_config": str_pretty_json(self._lr_scheduler_config),
                    "lr_scheduler_handler": self._lr_scheduler_handler,
                    "add_module_to_engine": self._add_module_to_engine,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def create(self, engine: BaseEngine, optimizer: Optimizer | None) -> LRSchedulerType | None:
        lr_scheduler = setup_lr_scheduler(
            optimizer=optimizer, lr_scheduler=self._lr_scheduler_config
        )
        if lr_scheduler is None:
            return None

        logger.info(f"lr_scheduler:\n{lr_scheduler}")
        if self._add_module_to_engine:
            logger.info(f"Adding a LR scheduler to the engine state (key: {ct.LR_SCHEDULER})...")
            engine.add_module(ct.LR_SCHEDULER, lr_scheduler)

        if self._lr_scheduler_handler:
            logger.info("Attaching a LR scheduler manager to the engine...")
            self._lr_scheduler_handler.attach(engine=engine)
        else:
            logger.warning(
                "No LR scheduler manager is set. If you do not use a LR scheduler manager, you "
                "need to manage 'manually' the LR scheduler"
            )
        return lr_scheduler
