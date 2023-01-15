__all__ = ["MultiKeysSeqMapSorterIterDataPipe"]

from collections.abc import Hashable, Iterator, Mapping, Sequence

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent


class MultiKeysSeqMapSorterIterDataPipe(IterDataPipe[Sequence[Mapping]]):
    r"""Implements a DataPipe to sort a sequence of mappings according to the
    values of multiple keys.

    Args:
        source_datapipe (``IterDataPipe``): Specifies the source
            ``IterDataPipe``.
        keys (sequence): Specifies the keys used to sort the mappings.
            The first key is the primary key, the second key is the
            secondary key, etc.
    """

    def __init__(self, source_datapipe: IterDataPipe[Sequence[Mapping]], keys: Sequence[Hashable]):
        self._source_datapipe = source_datapipe
        self._keys = tuple(keys)

    def __iter__(self) -> Iterator[list[Mapping]]:
        for sequence in self._source_datapipe:
            yield sorted(sequence, key=lambda item: tuple(item[key] for key in self._keys))

    def __len__(self) -> int:
        return len(self._source_datapipe)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  keys={self._keys},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )
