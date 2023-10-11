from __future__ import annotations

__all__ = ["EpochLRMonitor", "IterationLRMonitor"]

import logging
from typing import TYPE_CHECKING

from gravitorch.engines.events import (
    EngineEvents,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.optimizers.utils import get_learning_rate_per_group
from gravitorch.utils.events import GConditionalEventHandler
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class EpochLRMonitor(BaseHandler):
    r"""Implements a handler to monitor the learning rate (LR) of an
    optimizer every ``freq`` epochs.

    You should use the ``EpochOptimizerMonitor`` handler if you want
    to monitor more information about the optimizer.

    Args:
    ----
        event (str, optional): Specifies the epoch-based event when
            the learning rate should be capture.
            Default: ``'train_epoch_started'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the learning rate. Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import EpochLRMonitor
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = EpochLRMonitor()
        >>> handler
        EpochLRMonitor(freq=1, event=train_epoch_started)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_epoch_started")
    """

    def __init__(self, event: str = EngineEvents.TRAIN_EPOCH_STARTED, freq: int = 1) -> None:
        self._event = str(event)
        if freq < 1:
            raise ValueError(f"freq has to be greater than 0 (received: {freq:,})")
        self._freq = int(freq)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq}, event={self._event})"

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=GConditionalEventHandler(
                self.monitor,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the learning rate.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import EpochLRMonitor
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = EpochLRMonitor()
            >>> handler.monitor(engine)
        """
        if engine.optimizer:
            lrs = get_learning_rate_per_group(engine.optimizer)
            engine.log_metrics(
                {f"epoch/optimizer.group{i}.lr": lr for i, lr in lrs.items()},
                step=EpochStep(engine.epoch),
            )
        else:
            logger.info(
                "It is not possible to monitor the learning rate parameters because "
                "there is no optimizer"
            )


class IterationLRMonitor(BaseHandler):
    r"""Implements a handler to monitor the learning rate (LR) of an
    optimizer every ``freq`` iterations.

    You should use the ``IterationOptimizerMonitor`` handler if you
    want to monitor more information about the optimizer.

    Args:
    ----
        event (str, optional): Specifies the iteration-based event
            when the learning rate should be capture.
            Default: ``'train_iteration_started'``
        freq (int, optional): Specifies the iteration frequency used
            to monitor the learning rate. Default: ``10``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import IterationLRMonitor
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = IterationLRMonitor()
        >>> handler
        IterationLRMonitor(freq=10, event=train_iteration_started)
        >>> handler.attach(engine)
        >>> engine.trigger_event("train_iteration_started")
    """

    def __init__(self, event: str = EngineEvents.TRAIN_ITERATION_STARTED, freq: int = 10) -> None:
        self._event = str(event)
        if freq < 1:
            raise ValueError(f"freq has to be greater than 0 (received: {freq:,})")
        self._freq = int(freq)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq}, event={self._event})"

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=GConditionalEventHandler(
                self.monitor,
                condition=IterationPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the learning rate.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import IterationLRMonitor
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = IterationLRMonitor()
            >>> handler.monitor(engine)
        """
        if engine.optimizer:
            lrs = get_learning_rate_per_group(engine.optimizer)
            engine.log_metrics(
                {f"iteration/optimizer.group{i}.lr": lr for i, lr in lrs.items()},
                step=IterationStep(engine.iteration),
            )
        else:
            logger.info(
                "It is not possible to monitor the learning rate parameters because "
                "there is no optimizer"
            )
