__all__ = ["ModelParameterAnalyzer"]

from typing import Union

from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import to_events
from gravitorch.nn.utils.parameter_analysis import show_parameter_stats
from gravitorch.utils.events import VanillaEventHandler


class ModelParameterAnalyzer(BaseHandler):
    r"""Implements a handler to analyze the model parameter values.

    Args:
        events (str or tuple or list): Specifies the event(s) when to
            analyze the model parameter values. For example, it is
            usually a good idea to look at the model parameters at the
            beginning and end of the training.
            Default: ``('started', 'train_completed')``
        tablefmt (str, optional): Specifies the table format. You can
            find the valid formats at
            https://pypi.org/project/tabulate/. Default: ``'rst'``
    """

    def __init__(
        self,
        events: Union[str, tuple[str, ...], list[str]] = (
            EngineEvents.STARTED,
            EngineEvents.TRAIN_COMPLETED,
        ),
        tablefmt: str = "rst",
    ):
        self._events = to_events(events)
        self._tablefmt = tablefmt

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(events={self._events}, tablefmt={self._tablefmt})"

    def attach(self, engine: BaseEngine) -> None:
        for event in self._events:
            engine.add_event_handler(
                event,
                VanillaEventHandler(self.analyze, handler_kwargs={"engine": engine}),
            )

    def analyze(self, engine: BaseEngine) -> None:
        r"""Analyzes the model parameter values.

        Args:
            engine (``BaseEngine``): Specifies the engine with the
                model to analyze.
        """
        show_parameter_stats(module=engine.model, tablefmt=self._tablefmt)
