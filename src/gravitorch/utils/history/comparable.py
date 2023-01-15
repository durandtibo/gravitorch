r"""This module implements some comparable histories."""

__all__ = ["ComparableHistory", "MaxScalarHistory", "MinScalarHistory"]

from collections.abc import Iterable
from numbers import Number
from typing import Any, Optional, TypeVar

from gravitorch.utils.history.base import EmptyHistoryError
from gravitorch.utils.history.comparator import (
    BaseComparator,
    MaxScalarComparator,
    MinScalarComparator,
)
from gravitorch.utils.history.generic import GenericHistory

T = TypeVar("T")


class ComparableHistory(GenericHistory[T]):
    r"""Implements a comparable history.

    Args:
        name (str): Specifies the name of the history.
        comparator (BaseComparator): Specifies the comparator to use
            to find the best value.
        elements (iterable, optional): Specifies the initial elements.
            Each element is a tuple with the step and its associated
            value. Default: ``tuple()``
        max_size (int, optional): Specifies the maximum size
            of the history. Default: ``10``
        best_value (``T`` or ``None``, optional): Specifies the
            initial best value. If ``None``, the initial best value of
            the ``comparator`` is used. Default: ``None``
        improved (bool, optional): Indicates if the last value is the
            best value or not. Default: ``False``
    """

    def __init__(
        self,
        name: str,
        comparator: BaseComparator[T],
        elements: Iterable[tuple[Optional[int], T]] = tuple(),
        max_size: int = 10,
        best_value: Optional[T] = None,
        improved: bool = False,
    ):
        super().__init__(name=name, elements=elements, max_size=max_size)
        self._comparator = comparator
        self._best_value = best_value or self._comparator.get_initial_best_value()
        self._improved = bool(improved)

    # TODO: add to string

    def add_value(self, value: T, step: Optional[int] = None) -> None:
        self._improved = self.is_better(new_value=value, old_value=self._best_value)
        if self._improved:
            self._best_value = value
        super().add_value(value, step)

    def is_better(self, old_value: T, new_value: T) -> bool:
        r"""Indicates if the new value is better than the old value.

        Args:
            old_value: Specifies the old value to compare.
            new_value: Specifies the new value to compare.

        Returns:
            bool: ``True`` if the new value is better than the old
                value, otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.history import ComparableHistory
            >>> from gravitorch.utils.history.comparator import MaxScalarComparator
            >>> history = ComparableHistory('accuracy', comparator=MaxScalarComparator())
            >>> history.is_better(new_value=1, old_value=0)
            True
            >>> history.is_better(new_value=0, old_value=1)
            False
        """
        return self._comparator.is_better(new_value=new_value, old_value=old_value)

    def _get_best_value(self) -> T:
        if self.is_empty():
            raise EmptyHistoryError(
                "The history is empty so it is not possible to get the best value."
            )
        return self._best_value

    def _has_improved(self) -> bool:
        if self.is_empty():
            raise EmptyHistoryError("The history is empty.")
        return self._improved

    def is_comparable(self) -> bool:
        return True

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        config["max_size"] = self.max_size
        config["comparator"] = self._comparator
        return config

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._improved = state_dict["improved"]
        self._best_value = state_dict["best_value"]

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"improved": self._improved, "best_value": self._best_value})
        return state


class MaxScalarHistory(ComparableHistory[Number]):
    r"""A specific implementation to track the max value of a scalar history.

    This history uses the ``MaxScalarComparator`` to find the
    best value of the history.

    Args:
        name (str): Specifies the name of the history.
        elements (iterable, optional): Specifies the initial elements.
            Each element is a tuple with the step and its associated
            value. Default: ``tuple()``
        max_size (int, optional): Specifies the maximum size
            of the history. Default: ``10``
        best_value (float or int or ``None``, optional): Specifies the
            initial best value. If ``None``, the initial best value of
            the ``comparator`` is used. Default: ``None``
        improved (bool, optional): Indicates if the last value is the
            best value or not. Default: ``False``
    """

    def __init__(
        self,
        name: str,
        elements: Iterable[tuple[Optional[int], T]] = tuple(),
        max_size: int = 10,
        best_value: Optional[T] = None,
        improved: bool = False,
    ):
        super().__init__(
            name=name,
            comparator=MaxScalarComparator(),
            elements=elements,
            max_size=max_size,
            best_value=best_value,
            improved=improved,
        )

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        del config["comparator"]
        return config


class MinScalarHistory(ComparableHistory[Number]):
    r"""A specific implementation to track the min value of a scalar history.

    This history uses the ``MinScalarComparator`` to find the
    best value of the history.

    Args:
        name (str): Specifies the name of the history.
        elements (iterable, optional): Specifies the initial elements.
            Each element is a tuple with the step and its associated
            value. Default: ``tuple()``
        max_size (int, optional): Specifies the maximum size
            of the history. Default: ``10``
        best_value (float or int or ``None``, optional): Specifies the
            initial best value. If ``None``, the initial best value of
            the ``comparator`` is used. Default: ``None``
        improved (bool, optional): Indicates if the last value is the
            best value or not. Default: ``False``
    """

    def __init__(
        self,
        name: str,
        elements: Iterable[tuple[Optional[int], T]] = tuple(),
        max_size: int = 10,
        best_value: Optional[T] = None,
        improved: bool = False,
    ):
        super().__init__(
            name=name,
            comparator=MinScalarComparator(),
            elements=elements,
            max_size=max_size,
            best_value=best_value,
            improved=improved,
        )

    def config_dict(self) -> dict[str, Any]:
        config = super().config_dict()
        del config["comparator"]
        return config
