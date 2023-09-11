from __future__ import annotations

__all__ = [
    "SequentialDataPipeCreator",
    "SequentialCreatorIterDataPipeCreator",
]

from collections.abc import Sequence
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_sequence
from torch.utils.data import IterDataPipe

from gravitorch.creators.datapipe.base import (
    BaseDataPipeCreator,
    setup_datapipe_creator,
)
from gravitorch.datapipes import create_chained_datapipe

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine


class SequentialDataPipeCreator(BaseDataPipeCreator):
    r"""Implements a ``DataPipe`` creator to create a sequence of
    ``DataPipe``s from their configuration.

    Args:
    ----
        config (dict or sequence of dict): Specifies the configuration
            of the ``DataPipe`` object to create. See description
            of the ``create_sequential_iter_datapipe`` function to
            learn more about the expected values.

    Raises:
    ------
        ValueError if the ``DataPipe`` configuration sequence is
            empty.

    Example usage:

    .. code-block:: pycon

        >>> from torch.utils.data.datapipes.iter import IterableWrapper
        >>> from gravitorch.creators.datapipe import SequentialDataPipeCreator
        >>> # Create an IterDataPipe object using a single IterDataPipe object and no source input
        >>> creator = SequentialDataPipeCreator(
        ...     {
        ...         "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...         "iterable": [1, 2, 3, 4],
        ...     }
        ... )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Equivalent to
        >>> creator = SequentialDataPipeCreator(
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
        >>> creator = SequentialDataPipeCreator(
        ...     config={"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an IterDataPipe object using two IterDataPipe objects and no source input
        >>> creator = SequentialDataPipeCreator(
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
        >>> creator = SequentialDataPipeCreator(
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
        >>> creator = SequentialDataPipeCreator(
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
        self._config = config

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_sequence(self._config))}\n)"

    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe:
        return create_chained_datapipe(config=self._config, source_inputs=source_inputs)


class SequentialCreatorIterDataPipeCreator(BaseDataPipeCreator):
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
        ...     SequentialCreatorIterDataPipeCreator,
        ...     SequentialDataPipeCreator,
        ... )
        >>> # Create an IterDataPipe object using a single IterDataPipe creator and no source input
        >>> creator = SequentialCreatorIterDataPipeCreator(
        ...     [
        ...         SequentialDataPipeCreator(
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
        >>> creator = SequentialCreatorIterDataPipeCreator(
        ...     [
        ...         SequentialDataPipeCreator(
        ...             {"_target_": "torch.utils.data.datapipes.iter.IterableWrapper"},
        ...         ),
        ...     ]
        ... )
        >>> datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        >>> # Create an IterDataPipe object using two IterDataPipe creators and no source input
        >>> creator = SequentialCreatorIterDataPipeCreator(
        ...     [
        ...         SequentialDataPipeCreator(
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             },
        ...         ),
        ...         SequentialDataPipeCreator(
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
        >>> creator = SequentialCreatorIterDataPipeCreator(
        ...     creators=[
        ...         SequentialDataPipeCreator(
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
        >>> creator = SequentialCreatorIterDataPipeCreator(
        ...     [
        ...         SequentialDataPipeCreator(
        ...             {"_target_": "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         ),
        ...         SequentialDataPipeCreator(
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
    ) -> IterDataPipe:
        datapipe = self._creators[0].create(engine=engine, source_inputs=source_inputs)
        for creator in self._creators[1:]:
            datapipe = creator.create(engine=engine, source_inputs=(datapipe,))
        return datapipe
