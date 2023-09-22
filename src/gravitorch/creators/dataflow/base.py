from __future__ import annotations

__all__ = ["BaseDataFlowCreator", "is_dataflow_creator_config", "setup_dataflow_creator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from objectory import AbstractFactory
from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object

if TYPE_CHECKING:
    from gravitorch.dataflow import BaseDataFlow
    from gravitorch.engines import BaseEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseDataFlowCreator(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a dataflow creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataflow import IterableDataFlowCreator
        >>> creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
        >>> creator
        IterableDataFlowCreator(cache=False, length=5)
        >>> dataflow = creator.create()
        >>> dataflow
        IterableDataFlow(length=5)
    """

    @abstractmethod
    def create(self, engine: BaseEngine | None = None) -> BaseDataFlow[T]:
        r"""Create a dataflow.

        Args:
        ----
            engine (``gravitorch.engines.BaseEngine`` or ``None``,
                optional): Specifies an engine. Default: ``None``

        Returns:
        -------
            ``BaseDataFlow``: The created dataflow.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.creators.dataflow import IterableDataFlowCreator
            >>> creator = IterableDataFlowCreator((1, 2, 3, 4, 5))
            >>> dataflow = creator.create()
            >>> dataflow
            IterableDataFlow(length=5)
        """


def is_dataflow_creator_config(config: dict) -> bool:
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

        >>> from gravitorch.creators.dataflow import is_dataflow_creator_config
        >>> is_dataflow_creator_config(
        ...     {"_target_": "gravitorch.creators.dataflow.IterableDataFlowCreator"}
        ... )
        True
    """
    return is_object_config(config, BaseDataFlowCreator)


def setup_dataflow_creator(creator: BaseDataFlowCreator[T] | dict) -> BaseDataFlowCreator[T]:
    r"""Sets up the dataflow creator.

    The dataflow creator is instantiated from its configuration by
    using the ``BaseDataFlowCreator`` factory function.

    Args:
    ----
        creator (``BaseDataFlowCreator`` or dict): Specifies the
            dataflow creator or its configuration.

    Returns:
    -------
        ``BaseDataFlowCreator``: The instantiated dataflow creator.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.creators.dataflow import setup_dataflow_creator
        >>> creator = setup_dataflow_creator(
        ...     {
        ...         "_target_": "gravitorch.creators.dataflow.IterableDataFlowCreator",
        ...         "iterable": (1, 2, 3, 4, 5),
        ...     }
        ... )
        >>> creator
        IterableDataFlowCreator(cache=False, length=5)
    """
    if isinstance(creator, dict):
        logger.info(
            "Initializing the dataflow creator from its configuration... "
            f"{str_target_object(creator)}"
        )
        creator = BaseDataFlowCreator.factory(**creator)
    return creator
