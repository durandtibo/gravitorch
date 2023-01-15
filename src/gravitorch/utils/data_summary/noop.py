r"""This module implements a no-operation data summary."""

__all__ = ["NoOpDataSummary"]

from typing import TypeVar

from gravitorch.utils.data_summary.base import BaseDataSummary

T = TypeVar("T")


class NoOpDataSummary(BaseDataSummary[T]):
    r"""Implements a no-operation data summary."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def add(self, data: T) -> None:
        r"""Adds new data to the summary.

        Args:
            data: Specifies the data to add to the summary.
        """

    def reset(self) -> None:
        r"""Resets the data summary."""

    def summary(self) -> dict:
        return {}
