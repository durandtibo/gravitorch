from __future__ import annotations

__all__ = ["EngineEvents", "EpochPeriodicCondition", "IterationPeriodicCondition"]

from typing import TYPE_CHECKING, Any

from minevent import BaseCondition

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class EpochPeriodicCondition(BaseCondition):
    r"""Implements an epoch periodic condition.

    This condition is true every ``freq`` epochs.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.engines.events import EpochPeriodicCondition
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> condition = EpochPeriodicCondition(engine, freq=3)
        >>> engine.epoch
        -1
        >>> condition.evaluate()
        False
        >>> engine.increment_epoch()
        >>> condition.evaluate()
        True
        >>> engine.increment_epoch()
        >>> condition.evaluate()
        False
        >>> engine.increment_epoch()
        >>> condition.evaluate()
        False
        >>> engine.increment_epoch()
        >>> condition.evaluate()
        True
    """

    def __init__(self, engine: BaseEngine, freq: int) -> None:
        self._engine = engine
        self._freq = int(freq)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq:,}, epoch={self._engine.epoch:,})"

    @property
    def freq(self) -> int:
        r"""``int``: The frequency of the condition."""
        return self._freq

    def evaluate(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns
        -------
            bool: ``True`` if the condition is ``True``, otherwise
                ``False``.
        """
        return self._engine.epoch % self._freq == 0

    def equal(self, other: Any) -> bool:
        if isinstance(other, EpochPeriodicCondition):
            return self.freq == other.freq
        return False


class IterationPeriodicCondition(BaseCondition):
    r"""Implements an iteration periodic condition.

    This condition is true every ``freq`` iterations.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.engines.events import IterationPeriodicCondition
        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
        >>> condition = IterationPeriodicCondition(engine, freq=3)
        >>> engine.iteration
        -1
        >>> condition.evaluate()
        False
        >>> engine.increment_iteration()
        >>> condition.evaluate()
        True
        >>> engine.increment_iteration()
        >>> condition.evaluate()
        False
        >>> engine.increment_iteration()
        >>> condition.evaluate()
        False
        >>> engine.increment_iteration()
        >>> condition.evaluate()
        True
    """

    def __init__(self, engine: BaseEngine, freq: int) -> None:
        self._engine = engine
        self._freq = int(freq)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(freq={self._freq:,}, "
            f"iteration={self._engine.iteration:,})"
        )

    @property
    def freq(self) -> int:
        r"""``int``: The frequency of the condition."""
        return self._freq

    def evaluate(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns
        -------
            bool: ``True`` if the condition is ``True``, otherwise ``False``.
        """
        return self._engine.iteration % self._freq == 0

    def equal(self, other: Any) -> bool:
        if isinstance(other, IterationPeriodicCondition):
            return self.freq == other.freq
        return False


class EngineEvents:
    r"""Engine specific event names.

    Every engine should fire these events.
    """

    # Generic. These events are used in both ``train`` and ``eval``
    STARTED: str = "started"
    EPOCH_STARTED: str = "epoch_started"
    EPOCH_COMPLETED: str = "epoch_completed"
    COMPLETED: str = "completed"

    # Train
    TRAIN_STARTED: str = "train_started"
    TRAIN_EPOCH_STARTED: str = "train_epoch_started"
    TRAIN_ITERATION_STARTED: str = "train_iteration_started"
    TRAIN_FORWARD_COMPLETED: str = "train_forward_completed"
    TRAIN_BACKWARD_COMPLETED: str = "train_backward_completed"
    TRAIN_ITERATION_COMPLETED: str = "train_iteration_completed"
    TRAIN_EPOCH_COMPLETED: str = "train_epoch_completed"
    TRAIN_COMPLETED: str = "train_completed"

    # Eval
    EVAL_STARTED: str = "eval_started"
    EVAL_EPOCH_STARTED: str = "eval_epoch_started"
    EVAL_ITERATION_STARTED: str = "eval_iteration_started"
    EVAL_ITERATION_COMPLETED: str = "eval_iteration_completed"
    EVAL_EPOCH_COMPLETED: str = "eval_epoch_completed"
    EVAL_COMPLETED: str = "eval_completed"
