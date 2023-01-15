__all__ = ["BaseEpochMetric", "BaseStateEpochMetric"]

import logging
from typing import Optional, Union

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.models.metrics.base import BaseMetric
from gravitorch.models.metrics.state import BaseState, setup_state
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.exp_trackers import EpochStep

logger = logging.getLogger(__name__)


class BaseEpochMetric(BaseMetric):
    r"""Implements the base class of an epoch-wise metric.

    By default, this metric is reset at the beginning of the epoch and
    the result is computed at the end of the epoch.

    Child classes must implement the following methods:
        - ``forward``
        - ``reset``
        - ``value``

    Args:
        mode (str): Specifies the mode (e.g train or eval).
        name (str): Specifies the name of the metric.
    """

    def __init__(self, mode: str, name: str):
        super().__init__()
        self._mode = str(mode)
        self._name = str(name)
        self._metric_name = f"{self._mode}/{self._name}"

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to the provided engine.

        This method adds event handlers to the engine.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        if self._mode == ct.TRAIN:
            reset_event = EngineEvents.TRAIN_EPOCH_STARTED
            value_event = EngineEvents.TRAIN_EPOCH_COMPLETED
        else:
            reset_event = EngineEvents.EVAL_EPOCH_STARTED
            value_event = EngineEvents.EVAL_EPOCH_COMPLETED

        engine.add_event_handler(reset_event, VanillaEventHandler(self.reset))
        engine.add_event_handler(
            value_event, VanillaEventHandler(self.value, handler_kwargs={"engine": engine})
        )


class BaseStateEpochMetric(BaseEpochMetric):
    r"""Defines a base class to implement a metric where a lower value is better
    than a large value.

    Child classes must implement the following method:
        - ``forward``

    Args:
        mode (str): Specifies the mode (e.g ``'train'`` or ``'eval'``).
        name (str): Specifies the name of the metric. The name is used
            to log the metric results.
        state (``BaseState`` or dict): Specifies the metric state or
            its configuration.
    """

    def __init__(self, mode: str, name: str, state: Union[BaseState, dict]):
        super().__init__(mode=mode, name=name)
        self._state = setup_state(state)

    def extra_repr(self) -> str:
        return f"mode={self._mode},\nname={self._name},\nstate={self._state}"

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to the provided engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        super().attach(engine)
        for history in self._state.get_histories(f"{self._metric_name}_"):
            engine.add_history(history)

    def reset(self) -> None:
        r"""Resets the metric."""
        self._state.reset()

    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates the metric and log the results given all the predictions
        previously seen.

        Args:
            engine (``BaseEngine`` or ``None``, optional): Specifies
                the engine. This argument is required to log the
                results in the engine. Default: ``None``.

        Returns:
             dict: The results of the metric
        """
        results = self._state.value(f"{self._metric_name}_")
        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
        return results
