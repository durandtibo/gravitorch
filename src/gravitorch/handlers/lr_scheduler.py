from __future__ import annotations

__all__ = ["LRScheduler", "EpochLRScheduler", "IterationLRScheduler"]

from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping

from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.lr_monitor import EpochLRMonitor, IterationLRMonitor
from gravitorch.handlers.lr_scheduler_updater import (
    EpochLRSchedulerUpdater,
    IterationLRSchedulerUpdater,
)
from gravitorch.handlers.utils import setup_handler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class LRScheduler(BaseHandler):
    r"""Implements a handler to update a learning rate (LR) scheduler and
    monitor the LR value.

    Args:
    ----
        lr_scheduler_updater (``BaseHandler`` or dict): Specifies the
            learning rate scheduler updater or its configuration. The
            LR scheduler updater is responsible to update the LR
            scheduler.
        lr_monitor (``BaseHandler`` or dict): Specifies the learning
            rate monitor or its configuration.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import (
        ...     LRScheduler,
        ...     EpochLRMonitor,
        ...     EpochLRSchedulerUpdater,
        ... )
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = LRScheduler(
        ...     lr_scheduler_updater=EpochLRSchedulerUpdater(), lr_monitor=EpochLRMonitor()
        ... )
        >>> handler
        LRScheduler(
          (lr_monitor): EpochLRMonitor(freq=1, event=train_epoch_started)
          (lr_scheduler_updater): EpochLRSchedulerUpdater(event=train_epoch_completed)
        )
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_epoch_started")
        >>> engine.trigger_event("train_epoch_completed")
    """

    def __init__(
        self,
        lr_scheduler_updater: BaseHandler | dict,
        lr_monitor: BaseHandler | dict,
    ) -> None:
        self._lr_scheduler_updater = setup_handler(lr_scheduler_updater)
        self._lr_monitor = setup_handler(lr_monitor)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "lr_monitor": self._lr_monitor,
                    "lr_scheduler_updater": self._lr_scheduler_updater,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the handler to update a LR scheduler and monitor the
        LR value.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.
        """
        self._lr_scheduler_updater.attach(engine)
        self._lr_monitor.attach(engine)


class EpochLRScheduler(LRScheduler):
    r"""Implements a handler to update a learning rate (LR) scheduler at
    the end of each training epoch and monitor the LR value.

    This LR scheduler handler sets up:

        - an event handler to update the LR scheduler at the end of
            each training epoch
        - a LR monitor to log the learning rate value(s) at the
            beginning of each training epoch

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochLRScheduler
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochLRScheduler()
        >>> handler
        EpochLRScheduler(
          (lr_monitor): EpochLRMonitor(freq=1, event=train_epoch_started)
          (lr_scheduler_updater): EpochLRSchedulerUpdater(event=train_epoch_completed)
        )
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_epoch_started")
        >>> engine.trigger_event("train_epoch_completed")
    """

    def __init__(self) -> None:
        super().__init__(
            lr_scheduler_updater=EpochLRSchedulerUpdater(), lr_monitor=EpochLRMonitor()
        )


class IterationLRScheduler(LRScheduler):
    r"""Implements a handler to update a learning rate (LR) scheduler at
    the end of each training iteration and monitor the LR value.

    This LR scheduler handler sets up:

        - an event handler to update the LR scheduler at the end of
            each training iteration
        - a LR monitor to log the learning rate value(s) at the
            beginning of each training iteration

    Args:
    ----
        freq (int, optional): Specifies the iteration frequency used
            to monitor the learning rate. Default: ``10``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import IterationLRScheduler
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = IterationLRScheduler()
        >>> handler
        IterationLRScheduler(
          (lr_monitor): IterationLRMonitor(freq=10, event=train_iteration_started)
          (lr_scheduler_updater): IterationLRSchedulerUpdater(event=train_iteration_completed)
        )
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_iteration_started")
        >>> engine.trigger_event("train_iteration_completed")
    """

    def __init__(self, freq: int = 10) -> None:
        super().__init__(
            lr_scheduler_updater=IterationLRSchedulerUpdater(),
            lr_monitor=IterationLRMonitor(freq=freq),
        )
