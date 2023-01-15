r"""This module implements a data summary for continuous numerical values."""

__all__ = ["BaseContinuousDataSummary", "prepare_quantiles"]

from collections import deque
from typing import TypeVar, Union

import torch
from torch import Tensor

from gravitorch.utils.data_summary.base import BaseDataSummary, EmptyDataSummaryError

T = TypeVar("T")

DEFAULT_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class BaseContinuousDataSummary(BaseDataSummary[T]):
    r"""Implements a data summary for continuous numerical values.

    This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values

    A child class has to implement the ``add`` method.

    Args:
        max_size (int, optional): Specifies the maximum size used to
            store the last values because it may not be possible to
            store all the values. This parameter is used to compute
            the median and the quantiles. Default: ``10000``
        quantiles (``torch.Tensor`` or tuple or list, optional):
            Specifies a sequence of quantiles to compute, which must
            be between 0 and 1 inclusive. Default:
            ``(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)``
    """

    def __init__(
        self,
        max_size: int = 10000,
        quantiles: Union[Tensor, tuple[float, ...], list[float]] = DEFAULT_QUANTILES,
    ):
        self._sum = 0.0
        self._count = 0.0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._quantiles = prepare_quantiles(quantiles)
        # Store only the N last values to scale to large number of values.
        self._values = deque(maxlen=max_size)
        self.reset()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(max_size={self._values.maxlen:,}, "
            f"quantiles={self._quantiles.tolist()})"
        )

    def count(self) -> int:
        r"""Gets the number of values seen by the summary.

        Returns:
            int: The number of values seen by the summary.
        """
        return int(self._count)

    def max(self) -> float:
        r"""Gets the max value.

        Returns:
            float: The max value.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._max_value

    def mean(self) -> float:
        r"""Computes the mean value.

        This value is computed on all the values seen.

        Returns:
            float: The mean value.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._sum / self._count

    def median(self) -> float:
        r"""Computes the median value from the last values.

        If there are more values than the maximum window size, only the
        last values are used. Internally, this summary uses a deque to
        track the last values and the median value is computed on the
        values in the deque. The median is not unique for values with
        an even number of elements. In this case the lower of the two
        medians is returned.

        Returns:
            float: The median value from the last values.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return torch.tensor(list(self._values), dtype=torch.float).median().item()

    def min(self) -> float:
        r"""Gets the min value.

        Returns:
            float: The min value.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._min_value

    def quantiles(self) -> Tensor:
        r"""Computes the quantiles.

        If there are more values than the maximum size, only the last
        values are used. Internally, this summary uses a deque to
        track the last values and the quantiles are computed on the
        values in the deque.

        Returns:
            float: The standard deviation from the last values.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return torch.quantile(torch.tensor(tuple(self._values), dtype=torch.float), self._quantiles)

    def reset(self) -> None:
        r"""Resets the data summary."""
        self._sum = 0.0
        self._count = 0.0
        self._min_value = float("inf")
        self._max_value = -float("inf")
        self._values.clear()

    def std(self) -> float:
        r"""Computes the standard deviation.

        If there are more values than the maximum size, only the last
        values are used. Internally, this summary uses a deque to
        track the last values and the standard deviation is computed
        on the values in the deque.

        Returns:
            float: The standard deviation from the last values.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return torch.tensor(self._values, dtype=torch.float).std(dim=0).item()

    def sum(self) -> float:
        r"""Gets the sum value.

        Returns:
            float: The sum value.

        Raises:
            ``EmptyDataSummaryError`` if the summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        return self._sum

    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns:
            dict: The data descriptive summary.

        Raises:
            ``EmptyDataSummaryError`` is the data summary is empty.
        """
        if not self._count:
            raise EmptyDataSummaryError("The summary is empty")
        summary = {
            "count": self.count(),
            "sum": self.sum(),
            "mean": self.mean(),
            "median": self.median(),
            "std": self.std(),
            "max": self.max(),
            "min": self.min(),
        }
        summary.update(
            {
                f"quantile {quantile:.3f}": value.item()
                for quantile, value in zip(self._quantiles, self.quantiles())
            }
        )
        return summary


def prepare_quantiles(quantiles: Union[Tensor, tuple[float, ...], list[float]]) -> Tensor:
    r"""Prepares the quantiles to be comaptible with ``torch.quantile``.

    Args:
        quantiles (``torch.Tensor``, tuple, list): Specifies a sequence
            of quantiles to compute, which must be between 0 and 1
            inclusive.

    Returns:
        ``torch.Tensor`` of type float and shape ``(num_quantiles,)``:
            The prepare quantiles.
    """
    if isinstance(quantiles, (list, tuple)):
        quantiles = torch.tensor(quantiles)
    return torch.sort(quantiles)[0]
