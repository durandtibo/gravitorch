from __future__ import annotations

__all__ = ["BaseIterDataPipeCreator", "is_datapipe_creator_config", "setup_iter_datapipe_creator"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config
from torch.utils.data import IterDataPipe

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)


class BaseIterDataPipeCreator(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement an ``IterDataPipe`` creator.

    A ``IterDataPipe`` creator is responsible to create a single
    DataPipe.

    Note: it is possible to create an ``IterDataPipe`` object without
    using this class.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import SequentialIterDataPipeCreator
        >>> creator = SequentialIterDataPipeCreator(
        ...     [
        ...         {
        ...             "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...             "iterable": [1, 2, 3, 4],
        ...         }
        ...     ]
        ... )
        >>> creator
        SequentialIterDataPipeCreator(
          (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
        )
        >>> datapipe = creator.create()
        >>> tuple(datapipe)
        (1, 2, 3, 4)
    """

    @abstractmethod
    def create(
        self, engine: BaseEngine | None = None, source_inputs: Sequence | None = None
    ) -> IterDataPipe:
        r"""Creates an ``IterDataPipe`` object.

        Args:
        ----
            engine (``BaseEngine`` or ``None``, optional): Specifies
                an engine. The engine can be used to initialize the
                ``IterDataPipe`` by using the current epoch value.
                Default: ``None``
            source_inputs (sequence or ``None``): Specifies the first
                positional arguments of the source ``IterDataPipe``.
                This argument can be used to create a new
                ``IterDataPipe`` object, that takes existing
                ``IterDataPipe`` objects as input. See examples below
                to see how to use it. If ``None``, ``source_inputs``
                is set to an empty tuple. Default: ``None``

        Returns:
        -------
            ``IterDataPipe``: The created ``IterDataPipe`` object.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.creators.datapipe import SequentialIterDataPipeCreator
            >>> creator = SequentialIterDataPipeCreator(
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
    ``BaseIterDataPipeCreator``.

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
            for a ``BaseIterDataPipeCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import is_datapipe_creator_config
        >>> is_datapipe_creator_config(
        ...     {
        ...         "_target_": "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
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
    return is_object_config(config, BaseIterDataPipeCreator)


def setup_iter_datapipe_creator(creator: BaseIterDataPipeCreator | dict) -> BaseIterDataPipeCreator:
    r"""Sets up an ``IterDataPipe`` creator.

    The ``IterDataPipe`` creator is instantiated from its
    configuration by using the ``BaseDataPipeCreator`` factory
    function.

    Args:
    ----
        creator (``BaseIterDataPipeCreator`` or dict): Specifies the
            ``IterDataPipe`` creator or its configuration.

    Returns:
    -------
        ``BaseIterDataPipeCreator``: The instantiated ``IterDataPipe``
            creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datapipe import setup_iter_datapipe_creator
        >>> creator = setup_iter_datapipe_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
        ...         "config": [
        ...             {
        ...                 "_target_": "torch.utils.data.datapipes.iter.IterableWrapper",
        ...                 "iterable": [1, 2, 3, 4],
        ...             }
        ...         ],
        ...     }
        ... )
        >>> creator
        SequentialIterDataPipeCreator(
          (0): {'_target_': 'torch.utils.data.datapipes.iter.IterableWrapper', 'iterable': [1, 2, 3, 4]}
        )
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing a IterDataPipe creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseIterDataPipeCreator.factory(**creator)
    if not isinstance(creator, BaseIterDataPipeCreator):
        logger.warning(f"creator is not a `BaseIterDataPipeCreator` (received: {type(creator)})")
    return creator
