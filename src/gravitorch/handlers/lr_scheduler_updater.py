r"""This module contains handlers to update a learning rate (LR) scheduler."""

__all__ = [
    "EpochLRSchedulerUpdater",
    "MetricEpochLRSchedulerUpdater",
    "IterationLRSchedulerUpdater",
    "LRSchedulerUpdater",
    "MetricLRSchedulerUpdater",
]

import logging

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import VanillaEventHandler

logger = logging.getLogger(__name__)


class LRSchedulerUpdater(BaseHandler):
    r"""Implements a handler to update the learning rate (LR) scheduler at a
    given event.

    Args:
        event (str): Specifies the event when the learning rate
            scheduler is updated.
    """

    def __init__(self, event: str):
        self._event = str(event)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(event={self._event})"

    def attach(self, engine: BaseEngine) -> None:
        if engine.lr_scheduler is None:
            logger.info("No event is added to the engine because the LR scheduler is not defined")
            return
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(engine.lr_scheduler.step),
        )


class EpochLRSchedulerUpdater(LRSchedulerUpdater):
    r"""Implements a handler to update the learning rate (LR) scheduler at the
    end of each training epoch."""

    def __init__(self):
        super().__init__(event=EngineEvents.TRAIN_EPOCH_COMPLETED)


class IterationLRSchedulerUpdater(LRSchedulerUpdater):
    r"""Implements a handler to update the learning rate (LR) scheduler at the
    end of each training iteration."""

    def __init__(self):
        super().__init__(event=EngineEvents.TRAIN_ITERATION_COMPLETED)


class MetricLRSchedulerUpdater(BaseHandler):
    r"""Implements a handler to update the learning rate (LR) scheduler with a
    metric value at a given event.

    This updater was designed to be used with ``ReduceLROnPlateau``
    or any LR schedulers that use a metric value as input of the
    ``step`` method.

    Args:
        event (str): Specifies the event when the learning rate
            scheduler is updated. The event should happen after the
            metric was computed.
        metric_name (str, optional): Specifies the metric to use to
            control the LR scheduler. This metric should be accessible
            with the ``get_history`` method of the engine.
            Default: ``'eval/loss'``
    """

    def __init__(self, event: str, metric_name: str = f"{ct.EVAL}/loss"):
        self._event = event
        self._metric_name = metric_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, metric_name={self._metric_name})"
        )

    def attach(self, engine: BaseEngine) -> None:
        if engine.lr_scheduler is None:
            logger.info("No event is added to the engine because the LR scheduler is not defined")
            return
        # Note that it is not possible to use the LR scheduler `step` method because the `step`
        # method needs a metric value as input.
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=VanillaEventHandler(self.step, handler_kwargs={"engine": engine}),
        )

    def step(self, engine: BaseEngine) -> None:
        r"""Updates the LR scheduler with a metric value.

        Args:
            engine (``BaseEngine``): Specifies the engine with the LR
                scheduler to update and the metric used to update the
                LR scheduler.
        """
        engine.lr_scheduler.step(engine.get_history(self._metric_name).get_last_value())


class MetricEpochLRSchedulerUpdater(MetricLRSchedulerUpdater):
    r"""Implements a handler to update the learning rate (LR) scheduler with a
    metric value at the end of each epoch.

    This updater was designed to be used with ``ReduceLROnPlateau`` or
    any LR schedulers that use a metric value as input of the ``step``
    method. The ``step`` method of the LR scheduler is called at the
    ``"epoch_completed"`` event, so you need to use a metric that is
    updated before this event.

    Args:
        metric_name (str, optional): Specifies the metric to use to
            control the LR scheduler. This metric should be accessible
            with the ``get_history`` method of the engine.
            Default: ``'eval/loss'``
    """

    def __init__(self, metric_name: str = f"{ct.EVAL}/loss"):
        super().__init__(event=EngineEvents.EPOCH_COMPLETED, metric_name=metric_name)
