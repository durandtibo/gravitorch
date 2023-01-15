r"""This module defines the base comparator and some implementations."""

__all__ = [
    "BaseComparator",
    "ComparatorAllCloseOperator",
    "ComparatorEqualityOperator",
    "MaxScalarComparator",
    "MinScalarComparator",
]

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Union

from coola import (
    AllCloseTester,
    BaseAllCloseOperator,
    BaseAllCloseTester,
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseComparator(Generic[T], ABC):
    r"""Definition of the base comparator class."""

    @abstractmethod
    def equal(self, other: Any) -> bool:
        r"""Indicates if two comparators are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the comparators are equal,
                ``False`` otherwise.
        """

    @abstractmethod
    def get_initial_best_value(self) -> T:
        r"""Gets the initial best value.

        Returns:
            The initial best value.
        """

    @abstractmethod
    def is_better(self, old_value: T, new_value: T) -> bool:
        r"""Indicates if the new value is better than the old value.

        Args:
            old_value: Specifies the old value to compare.
            new_value: Specifies the new value to compare.

        Returns:
            bool: ``True`` if the new value is better than the old
                value, otherwise ``False``.
        """


class MaxScalarComparator(BaseComparator[Union[float, int]]):
    r"""Implementation of a max comparator for scalar value.

    This comparator can be used to find the maximum value between two
    scalar values.
    """

    def equal(self, other: Any) -> bool:
        return isinstance(other, MaxScalarComparator)

    def get_initial_best_value(self) -> float:
        return -float("inf")

    def is_better(self, old_value: Union[float, int], new_value: Union[float, int]) -> bool:
        return old_value <= new_value


class MinScalarComparator(BaseComparator[Union[float, int]]):
    r"""Implementation of a min comparator for scalar value.

    This comparator can be used to find the minimum value between two
    scalar values.
    """

    def equal(self, other: Any) -> bool:
        return isinstance(other, MinScalarComparator)

    def get_initial_best_value(self) -> float:
        return float("inf")

    def is_better(self, old_value: Union[float, int], new_value: Union[float, int]) -> bool:
        return new_value <= old_value


class ComparatorAllCloseOperator(BaseAllCloseOperator[BaseComparator]):
    r"""Implements an allclose operator for ``BaseComparator`` objects."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: BaseComparator,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseComparator):
            if show_difference:
                logger.info(f"object2 is not a `BaseComparator` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseComparator` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


class ComparatorEqualityOperator(BaseEqualityOperator[BaseComparator]):
    r"""Implements an equality operator for ``BaseComparator`` objects."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: BaseComparator,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not isinstance(object2, BaseComparator):
            if show_difference:
                logger.info(f"object2 is not a `BaseComparator` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`BaseComparator` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if not AllCloseTester.has_allclose_operator(BaseComparator):
    AllCloseTester.add_allclose_operator(
        BaseComparator, ComparatorAllCloseOperator()
    )  # pragma: no cover
if not EqualityTester.has_equality_operator(BaseComparator):
    EqualityTester.add_equality_operator(
        BaseComparator, ComparatorEqualityOperator()
    )  # pragma: no cover
