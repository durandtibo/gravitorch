from __future__ import annotations

__all__ = ["IterableDataStreamCreator"]

from collections.abc import Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, TypeVar

from gravitorch.creators.datastream.base import BaseDataStreamCreator
from gravitorch.datastreams.iterable import IterableDataStream
from gravitorch.utils.factory import setup_object
from gravitorch.utils.format import str_mapping

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class IterableDataStreamCreator(BaseDataStreamCreator[T]):
    r"""Implements a simple ``IterableDataFlow`` creator.

    Args:
    ----
        iterable (``Iterable`` or dict): Specifies an iterable or its
            configuration.
        cache (bool, optional): If ``True``, the iterable is created
            only the first time, and then a copy of the iterable is
            returned for each call to the ``create`` method.
            Default: ``False``
        **kwargs: See ``IterableDataFlow`` documentation.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datastream import IterableDataStreamCreator
        >>> creator = IterableDataStreamCreator((1, 2, 3, 4, 5))
        >>> creator
        IterableDataFlowCreator(cache=False, length=5)
        >>> datastream = creator.create()
        >>> datastream
        IterableDataFlow(length=5)
    """

    def __init__(self, iterable: Iterable[T], cache: bool = False, **kwargs) -> None:
        self._iterable = iterable
        self._cache = bool(cache)
        self._kwargs = kwargs

    def __repr__(self) -> str:
        config = {"cache": self._cache} | self._kwargs
        with suppress(TypeError):
            config["length"] = f"{len(self._iterable):,}"
        return (
            f"{self.__class__.__qualname__}({str_mapping(config, sorted_keys=True, one_line=True)})"
        )

    def create(self, engine: BaseEngine | None = None) -> IterableDataStream[T]:
        iterable = setup_object(self._iterable)
        if self._cache:
            self._iterable = iterable
        return IterableDataStream(iterable, **self._kwargs)
