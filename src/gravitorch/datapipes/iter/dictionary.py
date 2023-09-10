from __future__ import annotations

__all__ = ["DictOfListConverterIterDataPipe", "ListOfDictConverterIterDataPipe"]

from collections.abc import Hashable, Iterator, Mapping, Sequence

from coola.utils import str_indent, str_mapping
from torch.utils.data import IterDataPipe

from gravitorch.utils.mapping import convert_to_dict_of_lists, convert_to_list_of_dicts


class DictOfListConverterIterDataPipe(IterDataPipe[dict[Hashable, list]]):
    r"""Implements an ``IterDataPipe`` to convert a sequence of mappings
    to a dictionary of lists.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies an ``IterDataPipe``
            of sequences of mappings.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import DictOfListConverter
        >>> dp = DictOfListConverter(
        ...     IterableWrapper(
        ...         [
        ...             [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}],
        ...             [{"key": "a"}, {"key": -2}],
        ...         ]
        ...     ),
        ... )
        >>> dp
        DictOfListConverterIterDataPipe(
          (datapipe): IterableWrapperIterDataPipe
        )
        >>> list(dp)
        [{'key1': [1, 2, 3], 'key2': [10, 20, 30]}, {'key': ['a', -2]}]
    """

    def __init__(self, datapipe: IterDataPipe[Sequence[Mapping]]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[dict[Hashable, list]]:
        for data in self._datapipe:
            yield convert_to_dict_of_lists(data)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(str_mapping({"datapipe": self._datapipe}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"


class ListOfDictConverterIterDataPipe(IterDataPipe[list[dict]]):
    r"""Implements an ``IterDataPipe`` to convert a mapping of sequences
    to a list of dictionaries.

    Args:
    ----
        datapipe (``IterDataPipe``): Specifies an
            ``IterDataPipe`` of mappings of sequences.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.datapipes.iter import ListOfDictConverter
        >>> dp = ListOfDictConverter(
        ...     IterableWrapper([{"key1": [1, 2, 3], "key2": [10, 20, 30]}, {"key": ["a", -2]}]),
        ... )
        >>> dp
        ListOfDictConverterIterDataPipe(
          (datapipe): IterableWrapperIterDataPipe
        )
        >>> list(dp)
        [[{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}], [{'key': 'a'}, {'key': -2}]]
    """

    def __init__(self, datapipe: IterDataPipe[Mapping[Hashable, Sequence]]) -> None:
        self._datapipe = datapipe

    def __iter__(self) -> Iterator[list[dict]]:
        for data in self._datapipe:
            yield convert_to_list_of_dicts(data)

    def __len__(self) -> int:
        return len(self._datapipe)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        args = str_indent(str_mapping({"datapipe": self._datapipe}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"
