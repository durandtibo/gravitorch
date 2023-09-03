from __future__ import annotations

__all__ = ["BasePartitioner", "setup_partitioner"]

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from objectory import AbstractFactory

from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BasePartitioner(Generic[T], ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement a partitioner.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.partitioners import FixedSizePartitioner
        >>> partitioner = FixedSizePartitioner(partition_size=3)
        >>> partitioner
        FixedSizePartitioner(partition_size=3, drop_last=False)
        >>> partitions = partitioner.partition(list(range(10)))
        >>> partitions
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """

    @abstractmethod
    def partition(self, items: Sequence[T], engine: BaseEngine | None = None) -> list[Sequence[T]]:
        r"""Creates data.

        Args:
        ----
            items: Specifies the sequence to partition.
            engine (``BaseEngine`` or ``None``): Specifies an engine.
                Default: ``None``

        Return:
        ------
            list: The list of partitions.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.data.partitioners import FixedSizePartitioner
            >>> partitioner = FixedSizePartitioner(partition_size=3)
            >>> partitions = partitioner.partition(list(range(10)))
            >>> partitions
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        """


def setup_partitioner(partitioner: BasePartitioner | dict) -> BasePartitioner:
    r"""Sets up a partitioner.

    The partitioner is instantiated from its configuration by using
    the ``BasePartitioner`` factory function.

    Args:
    ----
        partitioner (``BasePartitioner`` or dict): Specifies the
            partitioner or its configuration.

    Returns:
    -------
        ``BasePartitioner``: The instantiated partitioner.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.data.partitioners import setup_partitioner
        >>> partitioner = setup_partitioner(
        ...     {
        ...         "_target_": "gravitorch.data.partitioners.FixedSizePartitioner",
        ...         "partition_size": 3,
        ...     }
        ... )
        >>> partitioner
        FixedSizePartitioner(partition_size=3, drop_last=False)
    """
    if isinstance(partitioner, dict):
        logger.info(
            "Initializing a partitioner from its configuration... "
            f"{str_target_object(partitioner)}"
        )
        partitioner = BasePartitioner.factory(**partitioner)
    return partitioner
