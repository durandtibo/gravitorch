r"""This module defines some conditions that can be used in the event
system."""
from __future__ import annotations

__all__ = ["EpochPeriodicCondition", "IterationPeriodicCondition", "PeriodicCondition"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class PeriodicCondition:
    r"""Implements a periodic condition.

    This condition is true every ``freq`` events.

    Args:
        freq (int): Specifies the frequency.
    """

    def __init__(self, freq: int):
        self._freq = int(freq)
        self._step = 0

    def __call__(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns:
            bool: ``True`` if the condition is ``True``, otherwise
                ``False``.
        """
        condition = self._step % self._freq == 0
        self._step += 1
        return condition

    def __eq__(self, other) -> bool:
        if isinstance(other, PeriodicCondition):
            return self.freq == other.freq
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq:,}, step={self._step:,})"

    @property
    def freq(self) -> int:
        r"""int: The frequency of the condition."""
        return self._freq


class EpochPeriodicCondition:
    r"""Implements an epoch periodic condition.

    This condition is true every ``freq`` epochs.

    Args:
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.
    """

    def __init__(self, engine: BaseEngine, freq: int):
        self._engine = engine
        self._freq = int(freq)

    def __call__(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns:
            bool: ``True`` if the condition is ``True``, otherwise
                ``False``.
        """
        return self._engine.epoch % self._freq == 0

    def __eq__(self, other) -> bool:
        if isinstance(other, EpochPeriodicCondition):
            return self.freq == other.freq
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq:,}, epoch={self._engine.epoch:,})"

    @property
    def freq(self) -> int:
        r"""int: The frequency of the condition."""
        return self._freq


class IterationPeriodicCondition:
    r"""Implements an iteration periodic condition.

    This condition is true every ``freq`` iterations.

    Args:
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.
    """

    def __init__(self, engine: BaseEngine, freq: int):
        self._engine = engine
        self._freq = int(freq)

    def __call__(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns:
            bool: ``True`` if the condition is ``True``, otherwise ``False``.
        """
        return self._engine.iteration % self._freq == 0

    def __eq__(self, other) -> bool:
        if isinstance(other, IterationPeriodicCondition):
            return self.freq == other.freq
        return False

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(freq={self._freq:,}, "
            f"iteration={self._engine.iteration:,})"
        )

    @property
    def freq(self) -> int:
        r"""int: The frequency of the condition."""
        return self._freq
