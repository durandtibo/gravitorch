r"""This module implements a data summary for discrete numerical values."""

__all__ = ["BaseDiscreteDataSummary"]

from collections import Counter
from typing import Optional, TypeVar

from gravitorch.utils.data_summary.base import BaseDataSummary, EmptyDataSummaryError

T = TypeVar("T")


class BaseDiscreteDataSummary(BaseDataSummary[T]):
    r"""Implements a data summary for discrete numerical values.

    This data summary assumes that the data are discrete numerical values.
    This data summary computes the following statistics:

        - ``count``: the number of values
        - ``num_unique_values``: the number of unique values
        - ``count_{}``: the number of values per unique value

    A child class has to implement the ``add`` method.
    """

    def __init__(self):
        self._counter = Counter()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def count(self) -> int:
        r"""Gets the number of values seen by the summary.

        Returns:
            int: The number of values seen by the summary.
        """
        return len(tuple(self._counter.elements()))

    def most_common(self, n: Optional[int] = None) -> list[tuple[int, int]]:
        r"""Gets a list of the ``n`` most common elements and their counts from
        the most common to the least.

        Args:
            n (int or None, optional): Specifies the number of elements to return.
                If ``n`` is ``None``, this method returns all elements in the counter.

        Returns:
            list: The list of the ``n`` most common elements and their counts.
                Elements with equal counts are ordered in the order first encountered.
        """
        if not self.count():
            raise EmptyDataSummaryError("The summary is empty")
        return self._counter.most_common(n)

    def reset(self) -> None:
        r"""Resets the data summary."""
        self._counter.clear()

    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns:
            dict: The data descriptive summary.

        Raises:
            ``EmptyDataSummaryError`` is the data summary is empty.
        """
        if not self.count():
            raise EmptyDataSummaryError("The summary is empty")
        summary = {
            "count": self.count(),
            "num_unique_values": len(self._counter),
        }
        for name, count in self.most_common():
            summary[f"count_{name}"] = count
        return summary
