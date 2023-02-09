__all__ = ["VanillaLRScheduler", "EpochLRScheduler", "IterationLRScheduler"]

from typing import Union

from gravitorch.engines.base import BaseEngine
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.lr_monitor import EpochLRMonitor, IterationLRMonitor
from gravitorch.handlers.lr_scheduler_updater import (
    EpochLRSchedulerUpdater,
    IterationLRSchedulerUpdater,
)
from gravitorch.handlers.utils import setup_handler
from gravitorch.utils.format import str_indent


class VanillaLRScheduler(BaseHandler):
    r"""Implements a handler to update a learning rate (LR) scheduler and
    monitor the LR value.

    Args:
        lr_scheduler_updater (``BaseHandler`` or dict): Specifies the
            learning rate scheduler updater or its configuration. The
            LR scheduler updater is responsible to update the LR
            scheduler.
        lr_monitor (``BaseHandler`` or dict): Specifies the learning
            rate monitor or its configuration.
    """

    def __init__(
        self,
        lr_scheduler_updater: Union[BaseHandler, dict],
        lr_monitor: Union[BaseHandler, dict],
    ):
        self._lr_scheduler_updater = setup_handler(lr_scheduler_updater)
        self._lr_monitor = setup_handler(lr_monitor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  lr_scheduler_updater={str_indent(str(self._lr_scheduler_updater))},\n"
            f"  lr_monitor={str_indent(str(self._lr_monitor))},\n"
            ")"
        )

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches the handler to update a LR scheduler and monitor the LR
        value.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        self._lr_scheduler_updater.attach(engine)
        self._lr_monitor.attach(engine)


class EpochLRScheduler(VanillaLRScheduler):
    r"""Implements a handler to update a learning rate (LR) scheduler at the end
    of each training epoch and monitor the LR value.

    This LR scheduler handler sets up:

        - an event handler to update the LR scheduler at the end of
            each training epoch
        - a LR monitor to log the learning rate value(s) at the
            beginning of each training epoch
    """

    def __init__(self):
        super().__init__(
            lr_scheduler_updater=EpochLRSchedulerUpdater(), lr_monitor=EpochLRMonitor()
        )


class IterationLRScheduler(VanillaLRScheduler):
    r"""Implements a handler to update a learning rate (LR) scheduler at the end
    of each training iteration and monitor the LR value.

    This LR scheduler handler sets up:

        - an event handler to update the LR scheduler at the end of
            each training iteration
        - a LR monitor to log the learning rate value(s) at the
            beginning of each training iteration

    Args:
        freq (int, optional): Specifies the iteration frequency used
            to monitor the learning rate. Default: ``10``
    """

    def __init__(self, freq: int = 10):
        super().__init__(
            lr_scheduler_updater=IterationLRSchedulerUpdater(),
            lr_monitor=IterationLRMonitor(freq=freq),
        )
