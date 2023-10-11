from __future__ import annotations

__all__ = ["ModelParameterAnalyzer"]

from typing import TYPE_CHECKING

from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import to_events
from gravitorch.nn.utils.parameter import show_parameter_summary
from gravitorch.utils.events import GEventHandler

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class ModelParameterAnalyzer(BaseHandler):
    r"""Implements a handler to analyze the model parameter values.

    Args:
    ----
        events (str or tuple or list): Specifies the event(s) when to
            analyze the model parameter values. For example, it is
            usually a good idea to look at the model parameters at the
            beginning and end of the training.
            Default: ``('started', 'train_completed')``
        tablefmt (str, optional): Specifies the table format. You can
            find the valid formats at
            https://pypi.org/project/tabulate/.
            Default: ``'fancy_outline'``
        floatfmt (str, optional): Specifies the float format.
            Default: ``'.6f'``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.handlers import ModelParameterAnalyzer
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> handler = ModelParameterAnalyzer()
        >>> handler
        ModelParameterAnalyzer(events=('started', 'train_completed'), tablefmt=fancy_outline, floatfmt=.6f)
        >>> handler.attach(engine)
        >>> engine.trigger_event("started")
    """

    def __init__(
        self,
        events: str
        | tuple[str, ...]
        | list[str] = (
            EngineEvents.STARTED,
            EngineEvents.TRAIN_COMPLETED,
        ),
        tablefmt: str = "fancy_outline",
        floatfmt: str = ".6f",
    ) -> None:
        self._events = to_events(events)
        self._tablefmt = str(tablefmt)
        self._floatfmt = str(floatfmt)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(events={self._events}, "
            f"tablefmt={self._tablefmt}, floatfmt={self._floatfmt})"
        )

    def attach(self, engine: BaseEngine) -> None:
        for event in self._events:
            engine.add_event_handler(
                event,
                GEventHandler(self.analyze, handler_kwargs={"engine": engine}),
            )

    def analyze(self, engine: BaseEngine) -> None:
        r"""Analyzes the model parameter values.

        Args:
        ----
            engine (``BaseEngine``): Specifies the engine with the
                model to analyze.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.handlers import ModelParameterAnalyzer
            >>> from gravitorch.testing import create_dummy_engine
            >>> engine = create_dummy_engine()
            >>> handler = ModelParameterAnalyzer()
            >>> handler.analyze(engine)
        """
        show_parameter_summary(
            module=engine.model, tablefmt=self._tablefmt, floatfmt=self._floatfmt
        )
