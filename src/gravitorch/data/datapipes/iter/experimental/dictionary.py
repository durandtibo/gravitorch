__all__ = ["RenameAllKeysIterDataPipe", "UpdateDictIterDataPipe"]

import logging
from collections.abc import Iterator

from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_add_indent

logger = logging.getLogger(__name__)


class RenameAllKeysIterDataPipe(IterDataPipe[dict]):
    r"""Implements a DataPipe that rename all the keys in a dictionary.

    Args:
        source_datapipe: Specifies the source iterable DataPipe.
        key_mapping (dict): Specifies the mapping between the old keys
            and the new keys. The keys (resp. values) of this
            dictionary indicates the original (resp. new) keys. The
            keys not present in the dictionary will be ignored.

    Example usage:

    .. code-block:: python

        >>> from torch.utils.data import IterDataPipe
        >>> from gravitorch.data.datapipes.iter.experimental import RenameAllKeysIterDataPipe
        >>> class MyIterDataPipe(IterDataPipe[dict]):
        ...     def __iter__(self) -> Iterator[dict]:
        ...         for i in range(3):
        ...             yield {"key1": 10 + i, "key2": 10 - i}
        ...
        >>> datapipe = RenameAllKeysIterDataPipe(
        ...     MyIterDataPipe(),
        ...     key_mapping={"key1": "new_key1", "key2": "new_key2"},
        ... )
        >>> for data in datapipe:
        ...     print(data)
        {'new_key1': 10, 'new_key2': 10}
        {'new_key1': 11, 'new_key2': 9}
        {'new_key1': 12, 'new_key2': 8}
    """

    def __init__(self, source_datapipe: IterDataPipe[dict], key_mapping: dict):
        self._source_datapipe = source_datapipe
        self._key_mapping = key_mapping

    def __iter__(self) -> Iterator[dict]:
        for sample in self._source_datapipe:
            yield {self._key_mapping[k]: v for k, v in sample.items()}

    def __len__(self) -> int:
        try:
            return len(self._source_datapipe)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__qualname__} instance doesn't have valid length"
            ) from exc

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  key_mapping={self._key_mapping},"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )


class UpdateDictIterDataPipe(IterDataPipe[dict]):
    r"""Implements an IterDataPipe to update dictionaries with an extra
    dictionary.

    Args:
        source_datapipe (``torch.utils.data.IterDataPipe``):
            Specifies the source iterable DataPipe.
        other (dict): Specifies the extra dictionary.
    """

    def __init__(self, source_datapipe: IterDataPipe[dict], other: dict):
        self._source_datapipe = source_datapipe
        self._other = other

    def __iter__(self) -> Iterator[dict]:
        for data in self._source_datapipe:
            data.update(self._other)
            yield data

    def __len__(self) -> int:
        try:
            return len(self._source_datapipe)
        except TypeError as exc:
            raise TypeError(
                f"{type(self).__qualname__} instance doesn't have valid length"
            ) from exc

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  other={self._other},\n"
            f"  source_datapipe={str_add_indent(self._source_datapipe)},\n)"
        )
