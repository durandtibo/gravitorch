r"""This module defines some conditions that can be used in the event
system."""
from __future__ import annotations

__all__ = ["EpochPeriodicCondition", "IterationPeriodicCondition"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class EpochPeriodicCondition:
    r"""Implements an epoch periodic condition.

    This condition is true every ``freq`` epochs.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.
    """

    def __init__(self, engine: BaseEngine, freq: int) -> None:
        self._engine = engine
        self._freq = int(freq)

    def __call__(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns
        -------
            bool: ``True`` if the condition is ``True``, otherwise
                ``False``.
        """
        return self._engine.epoch % self._freq == 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EpochPeriodicCondition):
            return self.freq == other.freq
        return False

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(freq={self._freq:,}, epoch={self._engine.epoch:,})"

    @property
    def freq(self) -> int:
        r"""``int``: The frequency of the condition."""
        return self._freq


class IterationPeriodicCondition:
    r"""Implements an iteration periodic condition.

    This condition is true every ``freq`` iterations.

    Args:
    ----
        engine (``BaseEngine``): Specifies the engine.
        freq (int): Specifies the frequency.
    """

    def __init__(self, engine: BaseEngine, freq: int) -> None:
        self._engine = engine
        self._freq = int(freq)

    def __call__(self) -> bool:
        r"""Evaluates the condition given the current state.

        Returns
        -------
            bool: ``True`` if the condition is ``True``, otherwise ``False``.
        """
        return self._engine.iteration % self._freq == 0

    def __eq__(self, other: Any) -> bool:
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
        r"""``int``: The frequency of the condition."""
        return self._freq
