from __future__ import annotations

__all__ = ["BaseDataPipeCreator", "is_datapipe_creator_config", "setup_datapipe_creator"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.utils.data import IterDataPipe, MapDataPipe

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataPipeCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a ``DataPipe`` creator.

    A ``DataPipe`` creator is responsible to create a single
    DataPipe.

    Note: it is possible to create a ``DataPipe`` object without
    using this class.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import ChainedDataPipeCreator
        >>> creator = ChainedDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         }
        ...     ]
        ... )
        >>> creator
        ChainedDataPipeCreator(
          (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
        )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe[T] | MapDataPipe[T]:
        r"""Creates a ``DataPipe`` object.

        Args:
        ----
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                ``DataPipe`` by using the current epoch value.
                Default: ``None``
            source_inputs (sequence or ``None``): Specifies the first
                positional arguments of the source ``DataPipe``.
                This argument can be used to create a new
                ``DataPipe`` object, that takes existing
                ``DataPipe`` objects as input. See examples below
                to see how to use it. If ``None``, ``source_inputs``
                is set to an empty tuple. Default: ``None``

        Returns:
        -------
            ``IterDataPipe`` or ``MapDataPipe``: The created
                ``DataPipe`` object.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.creators.datapipe import ChainedDataPipeCreator
            >>> creator = ChainedDataPipeCreator(
            ...     [
            ...         {
            ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
            ...             "iterable": [1, 2, 3, 4],
            ...         }
            ...     ]
            ... )
            >>> datapipe = creator.create()
            >>> tuple(datapipe)
            (1, 2, 3, 4)
        """


def is_datapipe_creator_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseDataPipeCreator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseDataPipeCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import is_datapipe_creator_config
        >>> is_datapipe_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.datapipe.ChainedDataPipeCreator",
        ...         "config": [
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             }
        ...         ],
        ...     }
        ... )
        True
    """
    return is_object_config(config, BaseDataPipeCreator)


def setup_datapipe_creator(creator: BaseDataPipeCreator | dict) -> BaseDataPipeCreator:
    r"""Sets up a ``DataPipe`` creator.

    The ``DataPipe`` creator is instantiated from its
    configuration by using the ``BaseDataPipeCreator`` factory
    function.

    Args:
    ----
        creator (``BaseDataPipeCreator`` or dict): Specifies the
            ``DataPipe`` creator or its configuration.

    Returns:
    -------
        ``BaseDataPipeCreator``: The instantiated ``DataPipe``
            creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import setup_datapipe_creator
        >>> creator = setup_datapipe_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.datapipe.ChainedDataPipeCreator",
        ...         "config": [
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             }
        ...         ],
        ...     }
        ... )
        >>> creator
        ChainedDataPipeCreator(
          (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a DataPipe creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataPipeCreator.factory(**creator)
    if not isinstance(creator, BaseDataPipeCreator):
        logger.warning(f"creator is not a `BaseDataPipeCreator` (received: {type(creator)})")
    return creator
