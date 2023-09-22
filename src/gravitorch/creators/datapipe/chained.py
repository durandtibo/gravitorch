from __future__ import annotations

__all__ = ["ChainedDataPipeCreator"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

from coola.utils import str_indent, str_sequence
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.creators.datapipe.base import BaseDataPipeCreator
from gravitorch.datapipes.factory import create_chained_datapipe

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

T = TypeVar("T")


class ChainedDataPipeCreator(BaseDataPipeCreator[T]):
    r"""Implements a ``DataPipe`` creator to create a sequence of
    ``DataPipe``s from their configuration.

    Args:
    ----
        config (dict or sequence of dict): Specifies the configuration
            of the ``DataPipe`` object to create. See description
            of the ``create_chained_datapipe`` function to
            learn more about the expected values.

    Raises:
    ------
        ValueError if the ``DataPipe`` configuration sequence is
            empty.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.datapipe import ChainedDataPipeCreator
        >>> # Create an IterDataPipe object using a single IterDataPipe object and no source input
        >>> creator = ChainedDataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Equivalent to
        >>> creator = ChainedDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # It is possible to use the source_inputs to create the same IterDataPipe object.
        >>> # The data is given by the source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config={"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an IterDataPipe object using two IterDataPipe objects and no source input
        >>> creator = ChainedDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         },
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ]
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to use the source_inputs to create the same IterDataPipe object.
        >>> # A source IterDataPipe object is specified by using source_inputs
        >>> creator = ChainedDataPipeCreator(
        ...     config=[
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ],
        ... )
        >>> datapipe = creator.create(source_inputs=[IterableWrapper([1, 2, 3, 4])])
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        >>> # It is possible to create a sequential ``IterDataPipe`` object that takes several
        >>> # IterDataPipe objects as input.
        >>> creator = ChainedDataPipeCreator(
        ...     config=[
        ...         {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.Batcher",
        ...             "batch_size": 2,
        ...         },
        ...     ],
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

    def __init__(self, config: dict | Sequence[dict]) -> None:
        if not config:
            raise ValueError("It is not possible to create a DataPipe because config is empty")
        if isinstance(config, dict):
            config = [config]
        self._config = tuple(config)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._config))}\n)"

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        return create_chained_datapipe(config=self._config, source_inputs=source_inputs)
