r"""This module implements a generic history."""
from __future__ import annotations

__all__ = ["GenericHistory"]

from collections import deque
from collections.abc import Iterable
from typing import Any, TypeVar

from coola import objects_are_equal

from gravitorch.utils.history.base import BaseHistory, EmptyHistoryError

T = TypeVar("T")


class GenericHistory(BaseHistory[T]):
    r"""Implements a generic history to store the recent values added in
    the history.

    Internally, this class uses a ``deque`` to keep the most recent
    values added in the history. Note that this class does not allow
    to get the best value because it is not possible to define a
    generic rule to know the best object. Please see
    ``ScalarHistory`` that can compute the best value for
    scalars.

    Args:
    ----
        name (str): Specifies the name of the history.
        elements (iterable, optional): Specifies the initial elements.
            Each element is a tuple with the step and its associated
            value. Default: ``tuple()``
        max_size (int, optional): Specifies the maximum size
            of the history. Default: ``10``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.history import GenericHistory
        >>> history = GenericHistory("value")
        >>> history.add_value(64.0)
        >>> history.add_value(42.0)
        >>> history.get_last_value()
        42.0
        >>> history.get_recent_history()
        ((None, 64.0), (None, 42.0))
    """

    def __init__(
        self,
        name: str,
        elements: Iterable[tuple[int | None, T]] = (),
        max_size: int = 10,
    ) -> None:
        super().__init__(name)
        if max_size <= 0:
            raise ValueError(f"History size must be greater than 0 (received: {max_size})")
        self._history = deque(elements, maxlen=max_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(name={self.name}, "
            f"max_size={self.max_size}, history={self.get_recent_history()})"
        )

    @property
    def max_size(self) -> int:
        r"""``int``: The maximum size of the history."""
        return self._history.maxlen

    def add_value(self, value: T, step: int | None = None) -> None:
        self._history.append((step, value))

    def equal(self, other: Any) -> bool:
        if not isinstance(other, GenericHistory):
            return False
        return objects_are_equal(self.to_dict(), other.to_dict())

    def get_last_value(self) -> Any:
        if self.is_empty():
            raise EmptyHistoryError(f"'{self.name}' history is empty.")
        return self._history[-1][1]

    def get_recent_history(self) -> tuple[tuple[int | None, T], ...]:
        return tuple(self._history)

    def is_comparable(self) -> bool:
        return False

    def is_empty(self) -> bool:
        return not self._history

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config["max_size"] = self.max_size
        return config

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._history = deque(state_dict["history"], maxlen=self.max_size)

    def state_dict(self) -> dict[str, Any]:
        return {"history": self.get_recent_history()}
