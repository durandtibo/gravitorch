r"""This module implements a data summary for integer values."""

__all__ = ["IntegerDataSummary"]

from gravitorch.utils.data_summary.discrete import BaseDiscreteDataSummary


class IntegerDataSummary(BaseDiscreteDataSummary[int]):
    r"""Implements a data summary for integer values.

    This data summary assumes that the data are discrete numerical
    values. This data summary computes the following statistics:

        - ``count``: the number of values
        - ``num_unique_values``: the number of unique values
        - ``count_{}``: the number of values per unique value
    """

    def add(self, data: int) -> None:
        r"""Adds new data to the summary.

        Args:
            data (int): Specifies the data to add to the summary.
        """
        self._counter[int(data)] += 1
