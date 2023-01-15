__all__ = ["FixedLengthIterDataPipe"]

from collections.abc import Iterator
from typing import TypeVar

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent

T = TypeVar("T")


class FixedLengthIterDataPipe(IterDataPipe[T]):
    r"""Implements a DataPipe that has a fixed length.

    - If the source DataPipe is longer than ``length``, only the first
        ``length`` items of the source DataPipe are used.
    - If the source DataPipe is shorter than ``length``, the items of
        the source DataPipe are repeated until ``length`` items.

    Args:
        source_datapipe (``torch.utils.data.IterDataPipe``): Specifies
            the source iterable DataPipe.
        length (int): Specifies the length of the DataPipe.
    """

    def __init__(self, source_datapipe: IterDataPipe[T], length: int):
        self._source_datapipe = source_datapipe
        if length < 1:
            raise ValueError(
                f"Incorrect length: {length}. The length has to be greater or equal to 1"
            )
        self._length = int(length)

    def __iter__(self) -> Iterator[T]:
        step = 0
        while step < self._length:
            for data in self._source_datapipe:
                yield data
                step += 1
                if step == self._length:
                    break

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  length={self._length:,},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )
