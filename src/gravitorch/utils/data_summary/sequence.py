r"""This module implements a base class to compute a descriptive summary of
sequences."""

__all__ = ["BaseSequenceDataSummary"]

from abc import abstractmethod
from typing import TypeVar

from gravitorch.utils.data_summary.base import BaseDataSummary
from gravitorch.utils.data_summary.float_value import FloatDataSummary

T = TypeVar("T")


class BaseSequenceDataSummary(BaseDataSummary[T]):
    r"""Implements a base class to compute a descriptive summary of sequences.

    A child class has to implement the ``_get_sequence_length`` method.

    Args:
        value_summary (``BaseDataSummary``): Specifies the summary
            object used to compute a descriptive summary of the
            values in the sequences.
    """

    def __init__(self, value_summary: BaseDataSummary[T]):
        self._value_summary = value_summary
        self._length_summary = FloatDataSummary()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  value_summary={self._value_summary},\n"
            f"  length_summary={self._length_summary}\n"
            ")"
        )

    def add(self, data: T) -> None:
        r"""Adds new data to the summary.

        Args:
            data: Specifies the data to add to the summary.
        """
        self._value_summary.add(data)
        self._length_summary.add(self._get_sequence_length(data))

    def reset(self) -> None:
        r"""Resets the data summary."""
        self._value_summary.reset()
        self._length_summary.reset()

    def summary(self) -> dict:
        r"""Gets a descriptive summary of the data.

        Returns:
            dict: The data descriptive summary. The dictionary has two
                keys: ``'length'`` and ``'value'``. The key
                ``'length'`` contains some information about the
                sequence length. The key ``'value'`` contains some
                information about the values in the sequence.

        Raises:
            ``EmptyDataSummaryError`` is the data summary is empty.
        """
        return {
            "value": self._value_summary.summary(),
            "length": self._length_summary.summary(),
        }

    @abstractmethod
    def _get_sequence_length(self, data: T) -> int:
        r"""Gets the sequence length of the data.

        Args:
            data : Specifies the input sequence.

        Returns:
            int: The sequence length.
        """
