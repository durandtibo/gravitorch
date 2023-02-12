__all__ = [
    "SequentialIterDataPipeCreator",
    "SequentialCreatorIterDataPipeCreator",
    "create_sequential_iter_datapipe",
]

from collections.abc import Sequence
from typing import Optional, Union

from objectory import OBJECT_TARGET, factory
from torch.utils.data import IterDataPipe

from gravitorch.creators.datapipe.base import (
    BaseIterDataPipeCreator,
    setup_iter_datapipe_creator,
)
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent, to_torch_sequence_str


class SequentialIterDataPipeCreator(BaseIterDataPipeCreator):
    r"""Implements an ``IterDataPipe`` creator to create a sequence of
    ``IterDataPipe``s from their configuration.

    Args:
        config (dict or sequence of dict): Specifies the configuration
            of the ``IterDataPipe`` object to create. See description
            of the ``create_sequential_iter_datapipe`` function to
            learn more about the expected values.

    Raises:
        ValueError if the ``IterDataPipe`` configuration sequence is
            empty.
    """

    def __init__(self, config: Union[dict, Sequence[dict]]):
        if not config:
            raise ValueError("It is not possible to create a DataPipe because `config` is empty")
        self._config = config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_sequence_str(self._config))},\n)"
        )

    def create(
        self, engine: Optional[BaseEngine] = None, source_inputs: Optional[Sequence] = None
    ) -> IterDataPipe:
        r"""Creates an ``IterDataPipe`` object.

        Args:
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine is not used by this creator.
                Default: ``None``
            source_inputs (sequence or ``None``): Specifies the first
                positional arguments of the source ``IterDataPipe``.
                This argument can be used to create a new
                ``IterDataPipe`` object, that takes existing
                ``IterDataPipe`` objects as input. See examples below
                to see how to use it. If ``None``, ``source_inputs`` is
                set to an empty tuple. Default: ``None``

        Returns:
            ``IterDataPipe``: The created ``IterDataPipe`` object.

        Example usage:

        .. code-block:: python

            >>> from torch.utils.data import IterDataPipe
            >>> from objectory import OBJECT_TARGET
            >>> from gravitorch.creators.datapipe import SequentialIterDataPipeCreator
            # Create an IterDataPipe object using a single IterDataPipe object and no source input
            >>> creator = SequentialIterDataPipeCreator(
            ...     {
            ...         OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            ...         "data": [1, 2, 3, 4],
            ...     }
            ... )
            >>> datapipe: IterDataPipe = creator.create()
            >>> tuple(datapipe)
            (1, 2, 3, 4)
            # Equivalent to
            >>> creator = SequentialIterDataPipeCreator(
            ...     [
            ...         {
            ...             OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            ...             "data": [1, 2, 3, 4],
            ...         },
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create()
            >>> tuple(datapipe)
            (1, 2, 3, 4)
            # It is possible to use the source_inputs to create the same IterDataPipe object.
            # The data is given by the source_inputs
            >>> creator = SequentialIterDataPipeCreator(
            ...     config={OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
            ... )
            >>> datapipe: IterDataPipe = creator.create(source_inputs=([1, 2, 3, 4],))
            >>> tuple(datapipe)
            (1, 2, 3, 4)
            # Create an IterDataPipe object using two IterDataPipe objects and no source input
            >>> creator = SequentialIterDataPipeCreator(
            ...     [
            ...         {
            ...             OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            ...             "data": [1, 2, 3, 4],
            ...         },
            ...         {
            ...             OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...             "batch_size": 2,
            ...         },
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create()
            >>> tuple(datapipe)
            ([1, 2], [3, 4])
            # It is possible to use the source_inputs to create the same IterDataPipe object.
            # A source IterDataPipe object is specified by using source_inputs
            >>> creator = SequentialIterDataPipeCreator(
            ...     config=[
            ...         {
            ...             OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...             "batch_size": 2,
            ...         },
            ...     ],
            ... )
            >>> from gravitorch.data.datapipes.iter import SourceWrapper
            >>> datapipe: IterDataPipe = creator.create(
            ...     source_inputs=[SourceWrapper(data=[1, 2, 3, 4])]
            ... )
            >>> tuple(datapipe)
            ([1, 2], [3, 4])
            # It is possible to create a sequential ``IterDataPipe`` object that takes several
            # IterDataPipe objects as input.
            >>> creator = SequentialIterDataPipeCreator(
            ...     config=[
            ...         {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            ...         {
            ...             OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...             "batch_size": 2,
            ...         },
            ...     ],
            ... )
            >>> datapipe: IterDataPipe = creator.create(
            ...    source_inputs=[
            ...         SourceWrapper(data=[1, 2, 3, 4]),
            ...         SourceWrapper(data=[11, 12, 13, 14]),
            ...     ],
            ... )
            >>> tuple(datapipe)
            ([1, 11], [2, 12], [3, 13], [4, 14])
        """
        return create_sequential_iter_datapipe(config=self._config, source_inputs=source_inputs)


def create_sequential_iter_datapipe(
    config: Union[dict, Sequence[dict]],
    source_inputs: Optional[Sequence] = None,
) -> IterDataPipe:
    r"""Creates a sequential ``IterDataPipe`` object.

    A sequential ``IterDataPipe`` object has a single source (which
    can takes multiple ``IterDataPipe`` objects) and a single sink.
    The structure should look like:

        SourceDatapipe -> DataPipe1 -> DataPipe2 -> SinkDataPipe

    The structure of the ``config`` input depends on the sequential
    ``IterDataPipe`` object that is created:

        - If ``config`` is a ``dict`` object, it creates a sequential
            ``IterDataPipe`` object with a single ``IterDataPipe``
            object. The dictionary should contain the parameters used
            to initialize the ``IterDataPipe`` object. It should
            follow the ``object_factory`` syntax. Using a dict allows
            to initialize a single ``IterDataPipe`` object. If you
            want to create a ``IterDataPipe`` object recursively, you
            need to give a sequence of dict.
        - If ``config`` is a sequence of ``dict`` objects, this
            function creates an ``IterDataPipe`` object with a
            sequential structure. The sequence of configurations
            follows the order of the ``IterDataPipe``s. The first
            config is used to create the first ``IterDataPipe``
            (a.k.a. source), and the last config is used to create the
            last ``IterDataPipe`` (a.k.a. sink). This function assumes
            all the DataPipes have a single source DataPipe as their
            first argument, excepts for the source ``IterDataPipe``.

    Note: it is possible to create sequential ``IterDataPipe`` objects
    without using this function.

    Args:
        config (dict or sequence of dict): Specifies the configuration
            of the ``IterDataPipe`` object to create. See description
            above to know when to use a dict or a sequence of dicts.
        source_inputs (sequence or ``None``): Specifies the first
            positional arguments of the source ``IterDataPipe``. This
            argument can be used to create a new ``IterDataPipe``
            object, that takes existing ``IterDataPipe`` objects as
            input. See examples below to see how to use it.
            If ``None``, ``source_inputs`` is set to an empty tuple.
            Default: ``None``

    Returns:
        ``IterDataPipe``: The last (a.k.a. sink) ``IterDataPipe`` of
            the sequence.

    Raises:
        ValueError if the configuration is empty (empty dict or
            sequence).

    Example usage:

    .. code-block:: python

        >>> from torch.utils.data import IterDataPipe
        >>> from objectory import OBJECT_TARGET
        >>> from gravitorch.creators.datapipe.sequential import create_sequential_iter_datapipe
        # Create an IterDataPipe object using a single IterDataPipe object and no source input
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     {
        ...         OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
        ...         "data": [1, 2, 3, 4],
        ...     }
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Equivalent to
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     [
        ...         {
        ...             OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...     ]
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # It is possible to use the source_inputs to create the same IterDataPipe object.
        # The data is given by the source_inputs
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     config={OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
        ...     source_inputs=([1, 2, 3, 4],),
        ... )
        >>> tuple(datapipe)
        (1, 2, 3, 4)
        # Create an IterDataPipe object using two IterDataPipe objects and no source input
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     [
        ...         {
        ...             OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
        ...             "data": [1, 2, 3, 4],
        ...         },
        ...         {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ]
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        # It is possible to use the source_inputs to create the same IterDataPipe object.
        # A source IterDataPipe object is specified by using source_inputs
        >>> from gravitorch.data.datapipes.iter import SourceWrapper
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     config=[
        ...         {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ],
        ...     source_inputs=[SourceWrapper(data=[1, 2, 3, 4])],
        ... )
        >>> tuple(datapipe)
        ([1, 2], [3, 4])
        # It is possible to create a sequential ``IterDataPipe`` object that takes several
        # IterDataPipe objects as input.
        >>> datapipe: IterDataPipe = create_sequential_iter_datapipe(
        ...     config=[
        ...         {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
        ...         {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ...     ],
        ...     source_inputs=[
        ...         SourceWrapper(data=[1, 2, 3, 4]),
        ...         SourceWrapper(data=[11, 12, 13, 14]),
        ...     ],
        ... )
        >>> tuple(datapipe)
        ([1, 11], [2, 12], [3, 13], [4, 14])
    """
    if not config:
        raise ValueError("It is not possible to create a DataPipe because `config` is empty")
    source_inputs = source_inputs or tuple()
    if isinstance(config, dict):
        config = config.copy()  # Make a copy because the dict is modified below.
        target = config.pop(OBJECT_TARGET)
        return factory(target, *source_inputs, **config)
    datapipe = create_sequential_iter_datapipe(config[0], source_inputs)
    for cfg in config[1:]:
        datapipe = create_sequential_iter_datapipe(cfg, source_inputs=(datapipe,))
    return datapipe


class SequentialCreatorIterDataPipeCreator(BaseIterDataPipeCreator):
    r"""Implements an ``IterDataPipe`` creator to create an ``IterDataPipe``
    object by using a sequence ``IterDataPipe`` creators.

    Args:
        creators: Specifies the sequence of ``IterDataPipe`` creators
            or their configurations. The sequence of creators follows
            the order of the ``IterDataPipe``s. The first creator is
            used to create the first ``IterDataPipe`` (a.k.a. source),
            and the last creator is used to create the last
            ``IterDataPipe`` (a.k.a. sink). This creator assumes all
            the DataPipes have a single source DataPipe as their first
            argument, excepts for the source ``IterDataPipe``.
    """

    def __init__(self, creators: Sequence[Union[BaseIterDataPipeCreator, dict]]):
        if not creators:
            raise ValueError("It is not possible to create a DataPipe because `creators` is empty")
        self._creators = [setup_iter_datapipe_creator(creator) for creator in creators]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  {str_indent(to_torch_sequence_str(self._creators))},\n)"
        )

    def create(
        self, engine: Optional[BaseEngine] = None, source_inputs: Optional[Sequence] = None
    ) -> IterDataPipe:
        r"""Creates an ``IterDataPipe`` object.

        Args:
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine.
                The engine is not used by this creator.
                Default: ``None``
            source_inputs (sequence or ``None``): Specifies the first
                positional arguments of the source ``IterDataPipe``.
                This argument can be used to create a new
                ``IterDataPipe`` object, that takes existing
                ``IterDataPipe`` objects as input.  See examples below
                to see how to use it. If ``None``, ``source_inputs``
                is setto an empty tuple. Default: ``None``

        Returns:
            ``IterDataPipe``: The created ``IterDataPipe`` object.

        Example usage:

        .. code-block:: python

            >>> from torch.utils.data import IterDataPipe
            >>> from objectory import OBJECT_TARGET
            >>> from gravitorch.creators.datapipe import (
            ...     SequentialCreatorIterDataPipeCreator,
            ...     SequentialIterDataPipeCreator,
            ... )
            # Create an IterDataPipe object using a single IterDataPipe creator and no source input
            >>> creator = SequentialCreatorIterDataPipeCreator(
            ...     [
            ...         SequentialIterDataPipeCreator(
            ...             {
            ...                 OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            ...                 "data": [1, 2, 3, 4],
            ...             },
            ...         ),
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create()
            >>> tuple(datapipe)
            (1, 2, 3, 4)
            # It is possible to use the source_inputs to create the same IterDataPipe object.
            # The data is given by the source_inputs
            >>> creator = SequentialCreatorIterDataPipeCreator(
            ...     [
            ...         SequentialIterDataPipeCreator(
            ...             {OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
            ...         ),
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create(source_inputs=([1, 2, 3, 4],))
            >>> tuple(datapipe)
            (1, 2, 3, 4)
            # Create an IterDataPipe object using two IterDataPipe creators and no source input
            >>> creator = SequentialCreatorIterDataPipeCreator(
            ...     [
            ...         SequentialIterDataPipeCreator(
            ...             {OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
            ...         ),
            ...         SequentialIterDataPipeCreator(
            ...             {
            ...                 OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...                 "batch_size": 2,
            ...             },
            ...         ),
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create()
            >>> tuple(datapipe)
            ([1, 2], [3, 4])
            # It is possible to use the source_inputs to create the same IterDataPipe object.
            # A source IterDataPipe object is specified by using source_inputs
            >>> from gravitorch.data.datapipes.iter import SourceWrapper
            >>> creator = SequentialCreatorIterDataPipeCreator(
            ...     creators=[
            ...         SequentialIterDataPipeCreator(
            ...             {
            ...                 OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...                 "batch_size": 2,
            ...             },
            ...         ),
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create(
            ...     source_inputs=[SourceWrapper(data=[1, 2, 3, 4])]
            ... )
            >>> tuple(datapipe)
            ([1, 2], [3, 4])
            # It is possible to create a sequential ``IterDataPipe`` object that takes several
            # IterDataPipe objects as input.
            >>> creator = SequentialCreatorIterDataPipeCreator(
            ...     [
            ...         SequentialIterDataPipeCreator(
            ...             {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            ...         ),
            ...         SequentialIterDataPipeCreator(
            ...             {
            ...                 OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
            ...                 "batch_size": 2,
            ...             },
            ...         ),
            ...     ]
            ... )
            >>> datapipe: IterDataPipe = creator.create(
            ...    source_inputs=[
            ...         SourceWrapper(data=[1, 2, 3, 4]),
            ...         SourceWrapper(data=[11, 12, 13, 14]),
            ...     ],
            ... )
            >>> tuple(datapipe)
            ([1, 11], [2, 12], [3, 13], [4, 14])
        """
        datapipe = self._creators[0].create(engine=engine, source_inputs=source_inputs)
        for creator in self._creators[1:]:
            datapipe = creator.create(engine=engine, source_inputs=(datapipe,))
        return datapipe
