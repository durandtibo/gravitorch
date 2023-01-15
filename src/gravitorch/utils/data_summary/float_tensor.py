r"""This module implements a data summary for ``torch.Tensor``s of type
float."""

__all__ = ["FloatTensorDataSummary", "FloatTensorSequenceDataSummary"]

from typing import Union

from torch import Tensor

from gravitorch.utils.data_summary.continuous import (
    DEFAULT_QUANTILES,
    BaseContinuousDataSummary,
)
from gravitorch.utils.data_summary.sequence import BaseSequenceDataSummary


class FloatTensorDataSummary(BaseContinuousDataSummary[Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type float.

    This data summary assumes that the data are continuous numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
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

    def add(self, data: Tensor) -> None:
        r"""Adds new data to the summary.

        Args:
            data (``torch.Tensor`` of type float): Specifies the data
                to add to the summary. This method converts the input
                to a ``torch.Tensor`` of type float if the tensor type
                is different.
        """
        if data.numel() > 0:
            values = data.float().flatten()
            self._sum += float(values.sum())
            self._count += values.numel()
            self._min_value = min(self._min_value, values.min().item())
            self._max_value = max(self._max_value, values.max().item())
            self._values.extend(values.tolist())


class FloatTensorSequenceDataSummary(BaseSequenceDataSummary[Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type float.

    The input should have a shape ``(sequence_length, *)`` where `*`
    means any number of dimensions. This data summary assumes that
    the data are continuous numerical values. This data summary
    computes the following statistics:

        - ``count``: the number of values
        - ``sum``: the sum of tne values
        - ``mean``: the mean of tne values
        - ``median``: the median of tne values
        - ``std``: the standard deviation of tne values
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

    def __init__(
        self,
        max_size: int = 10000,
        quantiles: Union[Tensor, tuple[float, ...], list[float]] = DEFAULT_QUANTILES,
    ):
        super().__init__(
            value_summary=FloatTensorDataSummary(max_size=max_size, quantiles=quantiles)
        )

    def _get_sequence_length(self, data: Tensor) -> int:
        r"""Gets the sequence length of the data.

        Args:
            data (``torch.Tensor`` of type float and shape
                ``(sequence_length, *)`` where `*` means any number of
                dimensions): Specifies the input sequence.

        Returns:
            int: The sequence length.
        """
        return data.shape[0]
