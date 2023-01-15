r"""This module implements a data summary for float values."""

__all__ = ["FloatDataSummary"]

from typing import TypeVar

from gravitorch.utils.data_summary.continuous import BaseContinuousDataSummary

T = TypeVar("T")

DEFAULT_QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class FloatDataSummary(BaseContinuousDataSummary[float]):
    r"""Implements a data summary for float values.

    This data summary assumes that the data are continuous numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of the values
        - ``max``: the max value
        - ``min``: the min value
        - ``quantiles``: the quantile values

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

    def add(self, data: float) -> None:
        r"""Adds new data to the summary.

        Args:
            data (float): Specifies the data to add to the summary.
        """
        value = float(data)
        self._sum += value
        self._count += 1.0
        self._min_value = min(self._min_value, value)
        self._max_value = max(self._max_value, value)
        self._values.append(value)
