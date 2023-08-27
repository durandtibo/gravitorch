from __future__ import annotations

__all__ = ["DictOfListConverterIterDataPipe", "ListOfDictConverterIterDataPipe"]

from collections.abc import Hashable, Iterator, Mapping, Sequence

from coola.utils import str_indent
from torch.utils.data import IterDataPipe

from gravitorch.utils.mapping import convert_to_dict_of_lists, convert_to_list_of_dicts


class DictOfListConverterIterDataPipe(IterDataPipe[dict[Hashable, list]]):
    r"""Implements an ``IterDataPipe`` to convert a sequence of mappings
    to a dictionary of lists.

    Args:
    ----
        source_datapipe (``IterDataPipe``): Specifies an ``IterDataPipe``
            of sequences of mappings.
    """

    def __init__(self, source_datapipe: IterDataPipe[Sequence[Mapping]]) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterator[dict[Hashable, list]]:
        for data in self._source_datapipe:
            yield convert_to_dict_of_lists(data)

    def __len__(self) -> int:
        return len(self._source_datapipe)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  source_datapipe={str_indent(self._source_datapipe)},\n)"
        )


class ListOfDictConverterIterDataPipe(IterDataPipe[list[dict]]):
    r"""Implements an ``IterDataPipe`` to convert a mapping of sequences
    to a list of dictionaries.

    Args:
    ----
        source_datapipe (``IterDataPipe``): Specifies an
            ``IterDataPipe`` of mappings of sequences.
    """

    def __init__(self, source_datapipe: IterDataPipe[Mapping[Hashable, Sequence]]) -> None:
        self._source_datapipe = source_datapipe

    def __iter__(self) -> Iterator[list[dict]]:
        for data in self._source_datapipe:
            yield convert_to_list_of_dicts(data)

    def __len__(self) -> int:
        return len(self._source_datapipe)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  source_datapipe={str_indent(self._source_datapipe)},\n)"
        )
