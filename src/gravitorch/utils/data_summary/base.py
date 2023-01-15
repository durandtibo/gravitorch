r"""This module implements the base class to implement a data summary."""

__all__ = ["BaseDataSummary", "EmptyDataSummaryError"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from objectory import AbstractFactory

T = TypeVar("T")


class BaseDataSummary(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a data summary.

    Note that the data summary only stores a descriptive summary of the
    data, and not the data.
    """

    @abstractmethod
    def add(self, data: T) -> None:
        r"""Adds new data to the summary.

        Args:
            data: Specifies the data to add to the summary.
        """

    @abstractmethod
    def reset(self) -> None:
        r"""Resets the data summary."""

    @abstractmethod
    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns:
            dict: The data descriptive summary.

        Raises:
            ``EmptyDataSummaryError`` is the data summary is empty.
        """


class EmptyDataSummaryError(Exception):
    r"""Raised when the data summary is empty because it is not possible to
    evaluate an empty data summary."""
