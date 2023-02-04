__all__ = ["EpochSysInfoMonitor"]

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import ConditionalEventHandler, EpochPeriodicCondition
from gravitorch.utils.sysinfo import log_system_info


class EpochSysInfoMonitor(BaseHandler):
    r"""Implements a handler to monitor the system metrics every ``freq``
    epochs.

    Args:
        event (str, optional): Specifies the epoch-based event when
            the system metrics should be capture.
            Default: ``'epoch_completed'``
        freq (int, optional): Specifies the epoch frequency used to
            monitor the system metrics. Default: ``1``
    """

    def __init__(self, event: str = EngineEvents.EPOCH_COMPLETED, freq: int = 1):
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
            event_handler=ConditionalEventHandler(
                self.monitor,
                condition=EpochPeriodicCondition(engine=engine, freq=self._freq),
                handler_kwargs={"engine": engine},
            ),
        )

    def monitor(self, engine: BaseEngine) -> None:
        r"""Monitors some system metrics.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        log_system_info()
