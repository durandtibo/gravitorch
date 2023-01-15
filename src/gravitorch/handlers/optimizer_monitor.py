__all__ = ["EpochOptimizerMonitor", "IterationOptimizerMonitor"]

import logging

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.optimizers.utils import (
    log_optimizer_parameters_per_group,
    show_optimizer_parameters_per_group,
)
from gravitorch.utils.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

logger = logging.getLogger(__name__)


class EpochOptimizerMonitor(BaseHandler):
    r"""Implements a handler to monitor the optimizer every ``freq`` epochs.

    Args:
        event (str, optional): Specifies the epoch-based event when
            the optimizer information should be capture.
            Default: ``'train_epoch_started'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the optimizer. Default: ``1``
        tablefmt (str, optional): Specifies the table format to show
            the optimizer information. You can find the valid formats
            at https://pypi.org/project/tabulate/.
            Default: ``'fancy_grid'``
        prefix (str, optional): Specifies the prefix which is used to
            log metrics. Default: ``"train/"``
    """

    def __init__(
        self,
        event: str = EngineEvents.TRAIN_EPOCH_STARTED,
        freq: int = 1,
        tablefmt: str = "fancy_grid",
        prefix: str = "train/",
    ):
        self._event = str(event)
        if freq < 1:
            raise ValueError(f"freq has to be greater than 0 (received: {freq:,})")
        self._freq = int(freq)
        self._tablefmt = str(tablefmt)
        self._prefix = str(prefix)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, freq={self._freq}, "
            f"tablefmt={self._tablefmt}, prefix={self._prefix})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=ConditionalEventHandler(
                self.monitor,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the current optimizer state.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        if engine.optimizer:
            show_optimizer_parameters_per_group(optimizer=engine.optimizer, tablefmt=self._tablefmt)
            log_optimizer_parameters_per_group(
                optimizer=engine.optimizer,
                engine=engine,
                step=EpochStep(engine.epoch),
                prefix=self._prefix,
            )
        else:
            logger.info(
                "It is not possible to monitor the optimizer parameters because there is no "
                "optimizer"
            )


class IterationOptimizerMonitor(BaseHandler):
    r"""Implements a handler to monitor the optimizer every ``freq`` iterations.

    Args:
        event (str, optional): Specifies the iteration-based event
            when the optimizer information should be capture.
            Default: ``'train_iteration_started'``
        freq (int, optional): Specifies the iteration frequency used
            to monitor the optimizer. Default: ``10``
        tablefmt (str, optional): Specifies the table format to show
            the optimizer information. You can find the valid formats
            at https://pypi.org/project/tabulate/.
            Default: ``'fancy_grid'``
        prefix (str, optional): Specifies the prefix which is used to
            log metrics. Default: ``"train/"``
    """

    def __init__(
        self,
        event: str = EngineEvents.TRAIN_ITERATION_STARTED,
        freq: int = 10,
        tablefmt: str = "fancy_grid",
        prefix: str = "train/",
    ):
        self._event = str(event)
        if freq < 1:
            raise ValueError(f"freq has to be greater than 0 (received: {freq:,})")
        self._freq = int(freq)
        self._tablefmt = str(tablefmt)
        self._prefix = str(prefix)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(event={self._event}, freq={self._freq}, "
            f"tablefmt={self._tablefmt}, prefix={self._prefix})"
        )

    def attach(self, engine: BaseEngine) -> None:
        add_unique_event_handler(
            engine=engine,
            event=self._event,
            event_handler=ConditionalEventHandler(
                self.monitor,
                condition=IterationPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors the current optimizer state.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        if engine.optimizer:
            show_optimizer_parameters_per_group(optimizer=engine.optimizer, tablefmt=self._tablefmt)
            log_optimizer_parameters_per_group(
                optimizer=engine.optimizer,
                engine=engine,
                step=IterationStep(engine.iteration),
                prefix=self._prefix,
            )
        else:
            logger.info(
                "It is not possible to monitor the optimizer parameters because there is no "
                "optimizer"
            )
