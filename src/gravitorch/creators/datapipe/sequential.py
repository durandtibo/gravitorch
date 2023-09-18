from __future__ import annotations

__all__ = ["SequentialDataPipeCreator"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_sequence
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.datapipe.base import (
    BaseDataPipeCreator,
    setup_datapipe_creator,
)

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class SequentialDataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements an ``IterDataPipe`` creator to create an
    ``IterDataPipe`` object by using a sequence ``IterDataPipe``
    creators.

    Args:
    ----
        creators: Specifies the sequence of ``IterDataPipe`` creators
            or their configurations. The sequence of creators follows
            the order of the ``IterDataPipe``s. The first creator is
            used to create the first ``IterDataPipe`` (a.k.a. source),
            and the last creator is used to create the last
            ``IterDataPipe`` (a.k.a. sink). This creator assumes all
            the DataPipes have a single source DataPipe as their first
            argument, excepts for the source ``IterDataPipe``.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.datapipe import (
        ...     SequentialDataPipeCreator,
        ...     ChainedDataPipeCreator,
        ... )
        >>> # Create an IterDataPipe object using a single IterDataPipe creator and no source input
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # It is possible to use the source_inputs to create the same IterDataPipe object.
        >>> # The data is given by the source_inputs
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an IterDataPipe object using two IterDataPipe creators and no source input
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             },
        ...         ),
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                 "batch_size": 2,
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to use the source_inputs to create the same IterDataPipe object.
        >>> # A source IterDataPipe object is specified by using source_inputs
        >>> creator = SequentialDataPipeCreator(
        ...     creators=[
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                 "batch_size": 2,
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to create a sequential ``IterDataPipe`` object that takes several
        >>> # IterDataPipe objects as input.
        >>> creator = SequentialDataPipeCreator(
        ...     [
        ...         ChainedDataPipeCreator(
        ...             {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         ),
        ...         ChainedDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...                 "batch_size": 2,
        ...             },
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(
        ...     source_inputs=[
        ...         IterableWrapper([1, 2, 3, 4]),
        ...         IterableWrapper([11, 12, 13, 14]),
        ...     ],
        ... )
        >>> tuple(datapipe)
        ([1, 11], [2, 12], [3, 13], [4, 14])
    """

    def __init__(self, creators: Sequence[BaseDataPipeCreator | dict]) -> None:
        if not creators:
            raise ValueError("It is not possible to create a DataPipe because creators is empty")
        self._creators = [setup_datapipe_creator(creator) for creator in creators]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._creators))}\n)"

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        datapipe = self._creators[0].create(engine=engine, source_inputs=source_inputs)
        for creator in self._creators[1:]:
            datapipe = creator.create(engine=engine, source_inputs=(datapipe,))
        return datapipe
