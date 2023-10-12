from __future__ import annotations

__all__ = ["BaseDataStreamCreator", "is_datastream_creator_config", "setup_datastream_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.datastreams import BaseDataStream
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataStreamCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a datastream creator.

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

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> BaseDataStream[T]:
        r"""Create a datastreams.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``BaseDataFlow``: The created datastreams.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.creators.datastream import IterableDataStreamCreator
            >>> creator = IterableDataStreamCreator((1, 2, 3, 4, 5))
            >>> datastream = creator.create()
            >>> datastream
            IterableDataFlow(length=5)
        """


def is_datastream_creator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseDataFlowCreator``.

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
            for a ``BaseDataFlowCreator`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datastream import is_datastream_creator_config
        >>> is_datastream_creator_config(
        ...     {"_target_": "gravitorch.creators.datastream.IterableDataFlowCreator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataStreamCreator)


def setup_datastream_creator(creator: BaseDataStreamCreator[T] | dict) -> BaseDataStreamCreator[T]:
    r"""Sets up the datastream creator.

    The datastream creator is instantiated from its configuration by
    using the ``BaseDataFlowCreator`` factory function.

    Args:
    ----
        creator (``BaseDataFlowCreator`` or dict): Specifies the
            datastream creator or its configuration.

    Returns:
    -------
        ``BaseDataFlowCreator``: The instantiated datastream creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.datastream import setup_datastream_creator
        >>> creator = setup_datastream_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.datastream.IterableDataFlowCreator",
        ...         "iterable": (1, 2, 3, 4, 5),
        ...     }
        ... )
        >>> creator
        IterableDataFlowCreator(cache=False, length=5)
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the datastream creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataStreamCreator.factory(**creator)
    return creator
