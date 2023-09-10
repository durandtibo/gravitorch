from __future__ import annotations

__all__ = ["LooperIterDataPipe"]

from collections.abc import Iterator
from typing import TypeVar

from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe

T = TypeVar("T")


class LooperIterDataPipe(IterDataPipe[T]):
    r"""Implements a DataPipe that has a fixed length.

    - If the source DataPipe is longer than ``length``, only the first
        ``length`` items of the source DataPipe are used.
    - If the source DataPipe is shorter than ``length``, the items of
        the source DataPipe are repeated until ``length`` items.

    Args:
    ----
        datapipe (``torch.utils.data.IterDataPipe``): Specifies
            the source iterable DataPipe.
        length (int): Specifies the length of the DataPipe.

     Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import Looper
        >>> dp = Looper(IterableWrapper([1, 2, 3, 4]), length=10)
        >>> dp
        LooperIterDataPipe(
          (length): 10
          (datapipe): IterableWrapperIterDataPipe
        )
        >>> list(dp)
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
        >>> dp = Looper(IterableWrapper([1, 2, 3, 4, 5, 6]), length=4)
        >>> list(dp)
        [1, 2, 3, 4]
    """

    def __init__(self, datapipe: IterDataPipe[T], length: int) -> None:
        self._datapipe = datapipe
        if length < 1:
            raise ValueError(
                f"Incorrect length: {length}. The length has to be greater or equal to 1"
            )
        self._length = int(length)

    def __iter__(self) -> Iterator[T]:
        step = 0
        while step < self._length:
            for data in self._datapipe:
                yield data
                step += 1
                if step == self._length:
                    break

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(str_mapping({"length": self._length, "datapipe": self._datapipe}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
