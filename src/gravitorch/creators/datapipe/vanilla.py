from __future__ import annotations

__all__ = ["DataPipeCreator"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.datapipe.base import BaseDataPipeCreator
from gravitorch.datapipes.factory import setup_datapipe
from gravitorch.datapipes.utils import clone_datapipe
from gravitorch.utils.format import str_indent, str_mapping

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class DataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements a simple ``DataPipe`` creator.

    Args:
    ----
        datapipe (``IterDataPipe`` or ``MapDataPipe`` or dict):
            Specifies the ``DataPipe`` or its configuration.
        cache (bool, optional): If ``True``, the ``DataPipe`` is
            created only the first time, and then the same
            ``DataPipe`` is returned for each call to the
            ``create`` method. Default: ``False``
        deepcopy (bool, optional): If ``True``, the ``DataPipe``
            object is deep-copied before to iterate over the data.
            It allows a deterministic behavior when in-place
            operations are performed on the data.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import DataPipeCreator
        >>> creator = DataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> creator
        DataPipeCreator(
          cache=False
          datapipe={'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
          deepcopy=False
        )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
    """

    def __init__(
        self,
        datapipe: IterDataPipe[T] | MapDataPipe[T] | dict,
        cache: bool = False,
        deepcopy: bool = False,
    ) -> None:
        self._datapipe = datapipe
        self._cache = bool(cache)
        self._deepcopy = bool(deepcopy)

    def __repr__(self) -> str:
        config = {"datapipe": self._datapipe, "cache": self._cache, "deepcopy": self._deepcopy}
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(str_mapping(config, sorted_keys=True))}\n)"
        )

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        datapipe = setup_datapipe(self._datapipe)
        if self._cache:
            self._datapipe = datapipe
        if self._deepcopy:
            datapipe = clone_datapipe(datapipe, raise_error=False)
        return datapipe
