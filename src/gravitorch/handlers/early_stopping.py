__all__ = ["EarlyStopping"]

import logging
import operator
from collections.abc import Callable

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.engines.events import EngineEvents
from gravitorch.handlers.base import BaseHandler
from gravitorch.handlers.utils import add_unique_event_handler
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import BaseHistory, MaxScalarHistory, MinScalarHistory

logger = logging.getLogger(__name__)


class EarlyStopping(BaseHandler):
    r"""Implements an early stopping handler to stop the training if no
    improvement after a given number of epochs.

    This early stopping handler only works with
    ``MaxScalarHistory`` and ``MinScalarHistory``
    metrics.

    Args:
        metric_name (str, optional): Specifies the metric name that
            is used to measure the improvement. By default, the metric
            is the loss on the evaluation data set.
            Default: 'eval/loss'.
        patience (int, optional): Specifies the number of epochs to
            wait if no improvement and then stop the training.
            Default: ``5``
        delta (float, optional): A minimum increase or decrease in
            the score to qualify as an improvement, i.e. an increase
            of less than or equal to `delta`, will count as no
            improvement. Default: ``0.0``
        cumulative_delta (bool, optional): It True, `min_delta`
            defines an increase since the last `patience` reset,
            otherwise, it defines an increase after the last epoch.
            Default: ``False``.

    Example usage:

    .. code-block:: python

        # Create an engine
        >>> engine = ...
        >>> from gravitorch.handlers import EarlyStopping
        # Add the early stopping handler to the engine with a patience of 10 epochs
        # for the accuracy metric.
        >>> handler = EarlyStopping(metric_name='eval/accuracy', patience=10)
        >>> handler.attach(engine)
    """

    def __init__(
        self,
        metric_name: str = f"{ct.EVAL}/loss",
        patience: int = 5,
        delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        self._metric_name = str(metric_name)
        if patience < 1:
            raise ValueError(f"Argument patience should be positive integer (received: {patience})")
        self._patience = int(patience)
        if delta < 0.0:
            raise ValueError(
                f"Argument 'min_delta' should not be a negative number  (received: {patience})"
            )
        self._delta = float(delta)
        self._cumulative_delta = bool(cumulative_delta)

        self._waiting_counter = 0
        self._best_score = None
        self._best_epoch = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(metric_name={self._metric_name}, "
            f"patience={self._patience}, "
            f"delta={self._delta}, "
            f"cumulative_delta={self._cumulative_delta}, "
            f"waiting_counter={self._waiting_counter}, "
            f"best_score={self._best_score}, "
            f"best_epoch={self._best_epoch})"
        )

    def attach(self, engine: BaseEngine) -> None:
        logger.info(
            f"Attach early stopping handler for metric {self._metric_name} with the parameters: "
            f"patience={self._patience:,}   delta={self._delta}   "
            f"cumulative_delta={self._cumulative_delta}"
        )
        add_unique_event_handler(
            engine=engine,
            event=EngineEvents.TRAIN_STARTED,
            event_handler=VanillaEventHandler(self.start, handler_kwargs={"engine": engine}),
        )
        add_unique_event_handler(
            engine=engine,
            event=EngineEvents.EPOCH_COMPLETED,
            event_handler=VanillaEventHandler(self.step, handler_kwargs={"engine": engine}),
        )

        # Add the module to the engine so the state of the early stopping module is saved when
        # the engine checkpoint is created. It is necessary to store the state to resume
        # the training.
        engine.add_module(ct.EARLY_STOPPING, self)

        if engine.has_history(self._metric_name):
            self._check_history(engine.get_history(self._metric_name))
        else:
            logger.warning(
                f"There is no history for '{self._metric_name}'. The history needs to be created "
                f"before the `step` method is called"
            )

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the state values from a dict.

        Args:
            state_dict (dict): a dict with parameters

        Example:

        .. code-block:: python

            >>> handler = EarlyStopping(...)
            >>> state = {...}
            >>> handler.load_state_dict(state)
        """
        self._best_epoch = state_dict["best_epoch"]
        self._best_score = state_dict["best_score"]
        self._waiting_counter = state_dict["waiting_counter"]

    def state_dict(self) -> dict:
        r"""Gets a dictionary containing state values.

        Returns:
            dict: the state values in a dict.

        Example:

        .. code-block:: python

            >>> state = EarlyStopping(...)
            >>> state_dict = state.state_dict()
        """
        return {
            "best_epoch": self._best_epoch,
            "best_score": self._best_score,
            "waiting_counter": self._waiting_counter,
        }

    def start(self, engine: BaseEngine) -> None:
        r"""Stops the training if the requirements to stop the training are
        already met.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        logger.info(
            f"Early stopping [start] | {self._metric_name} | "
            f"best_epoch {self._best_epoch} | best_score {self._best_score} | "
            f"waiting_counter {self._waiting_counter}/{self._patience}"
        )
        if self._waiting_counter >= self._patience:
            logger.info(
                f"Early stopping [start] | cancel training because '{self._metric_name}' did not "
                f"improve during the last {self._waiting_counter} epochs"
            )
            engine.terminate()

    def step(self, engine: BaseEngine) -> None:
        """Updates the early stopping handler by using the last value of the
        monitored metric.

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        history = engine.get_history(self._metric_name)
        current_score = history.get_last_value()
        min_delta = self._get_min_delta(history)
        comparator = self._get_comparator(history)

        if self._best_score is None:
            # The best score is undefined at the first call.
            # The first score becomes the best score.
            self._best_score = current_score
            self._best_epoch = engine.epoch

        elif not comparator(current_score - min_delta, self._best_score):
            # The current score is not better than the best score + min delta.
            self._waiting_counter += 1
            if not self._cumulative_delta and comparator(current_score, self._best_score):
                self._best_score = current_score
            if self._waiting_counter >= self._patience:
                logger.info(
                    f"Early stopping [step] | stop training because '{self._metric_name}' did not "
                    f"improve during the last {self._waiting_counter:,} epochs"
                )
                engine.terminate()

        else:
            # The current score is better than the best score.
            self._best_score = current_score
            self._waiting_counter = 0
            self._best_epoch = engine.epoch

        logger.info(
            f"Early stopping [step] | {self._metric_name} | "
            f"best_epoch {self._best_epoch} | best_score {self._best_score} | "
            f"waiting_counter {self._waiting_counter}/{self._patience}"
        )

    def _check_history(self, history: BaseHistory) -> None:
        r"""Checks the history.

        Args:
            history (``BaseHistory``): Specifies the history
                tracker to check.

        Raises:
            RuntimeError if the history tracker is not valid.
        """
        if not isinstance(history, (MaxScalarHistory, MinScalarHistory)):
            raise RuntimeError(
                f"The early stopping handler only supports ``MaxScalarHistory`` "
                f"or ``MinScalarHistory`` history tracker (received: {history})"
            )

    def _get_comparator(self, history: BaseHistory) -> Callable:
        r"""Gets the comparator function from the history tracker type.

        Args:
            history (``BaseHistory``): Specifies the history
                tracker.

        Returns:
            callable: The comparator.
        """
        if isinstance(history, MaxScalarHistory):
            return operator.gt  # >
        return operator.lt  # <

    def _get_min_delta(self, history: BaseHistory) -> float:
        r"""Gets the minimum delta.

        Args:
            history (``BaseHistory``): Specifies the history
                tracker.

        Returns:
            float: The minimum delta.
        """
        self._check_history(history)
        return self._delta if isinstance(history, MaxScalarHistory) else -self._delta
