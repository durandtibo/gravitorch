r"""This module implements a data summary for ``torch.Tensor``s of type
long."""

__all__ = ["LongTensorDataSummary", "LongTensorSequenceDataSummary"]

from torch import Tensor

from gravitorch.utils.data_summary.discrete import BaseDiscreteDataSummary
from gravitorch.utils.data_summary.sequence import BaseSequenceDataSummary


class LongTensorDataSummary(BaseDiscreteDataSummary[Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type long.

    This data summary assumes that the data are discrete numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``num_unique_values``: the number of unique values
        - ``count_{}``: the number of values per unique value
    """

    def add(self, data: Tensor) -> None:
        r"""Adds new data to the summary.

        Args:
            data (``torch.Tensor`` of type long): Specifies the data
                to add to the summary. This method converts the input
                to a ``torch.Tensor`` of type long if the tensor type
                is different.
        """
        if data.numel() > 0:
            self._counter.update(data.long().flatten().tolist())


class LongTensorSequenceDataSummary(BaseSequenceDataSummary[Tensor]):
    r"""Implements a data summary for ``torch.Tensor``s of type long.

    The input should have a shape ``(sequence_length, *)`` where `*`
    means any number of dimensions. This data summary assumes that
    the data are discrete numerical values. This data summary
    computes the following statistics:

        - ``count``: the number of values
        - ``num_unique_values``: the number of unique values
        - ``count_{}``: the number of values per unique value
    """

    def __init__(self):
        super().__init__(value_summary=LongTensorDataSummary())

    def _get_sequence_length(self, data: Tensor) -> int:
        r"""Gets the sequence length of the data.

        Args:
            data (``torch.Tensor`` of type long and shape
                ``(sequence_length, *)`` where `*` means any number of
                dimensions): Specifies the input sequence.

        Returns:
            int: The sequence length.
        """
        return data.shape[0]
