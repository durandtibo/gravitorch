from __future__ import annotations

__all__ = ["ModelArchitectureAnalyzer", "NetworkArchitectureAnalyzer"]

from typing import TYPE_CHECKING

from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import to_events
from gravitorch.models.utils import (
    analyze_model_architecture,
    analyze_network_architecture,
)
from gravitorch.utils.events import GEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class ModelArchitectureAnalyzer(BaseHandler):
    r"""Implements a handler to analyze a model architecture.

    Args:
    ----
        events (str or tuple or list): Specifies the event(s) when to
            analyze the model architecture. It is usually a good idea
            to log model information at the beginning of the training.
            Default: ``('started',)``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import ModelArchitectureAnalyzer
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = ModelArchitectureAnalyzer()
        >>> handler
        ModelArchitectureAnalyzer(events=('started',))
        >>> handler.attach(engine)
        >>> engine.trigger_event("started")
    """

    def __init__(
        self,
        events: str | tuple[str, ...] | list[str] = (EngineEvents.STARTED,),
    ) -> None:
        self._events = to_events(events)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(events={self._events})"

    def attach(self, engine: BaseEngine) -> None:
        for event in self._events:
            engine.add_event_handler(
                event,
                GEventHandler(self.analyze, handler_kwargs={"engine": engine}),
            )

    def analyze(self, engine: BaseEngine) -> None:
        r"""Analyzes the model architecture.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine with the
                model to analyze.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import ModelArchitectureAnalyzer
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = ModelArchitectureAnalyzer()
            >>> handler.analyze(engine)
        """
        analyze_model_architecture(model=engine.model, engine=engine)


class NetworkArchitectureAnalyzer(ModelArchitectureAnalyzer):
    r"""Implements a handler to analyze the network architecture of a
    model.

    This handler assumes the model has an attribute ``network``.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import NetworkArchitectureAnalyzer
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = NetworkArchitectureAnalyzer()
        >>> handler
        NetworkArchitectureAnalyzer(events=('started',))
        >>> handler.attach(engine)
        >>> engine.trigger_event("started")
    """

    def analyze(self, engine: BaseEngine) -> None:
        r"""Analyzes the network architecture of a model.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine with the
                model to analyze.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import NetworkArchitectureAnalyzer
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = NetworkArchitectureAnalyzer()
            >>> handler.analyze(engine)
        """
        analyze_network_architecture(model=engine.model, engine=engine)
