__all__ = ["DDPPartitioner", "SyncParallelPartitioner"]

from collections.abc import Sequence
from typing import Optional, TypeVar, Union

from gravitorch.data.partitioners.base import BasePartitioner, setup_partitioner
from gravitorch.distributed.ddp import broadcast_object_list
from gravitorch.engines.base import BaseEngine
from gravitorch.utils.format import str_indent
from gravitorch.utils.partitioning import ddp_partitions

T = TypeVar("T")


class DDPPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner designed for a Distributed Data Parallel (DDP)
    setting.

    All the partitions have the same number of items. The number of
    partitions is the world size of the distributed system.

    Args:
        partition_size (int): Specifies the partition size i.e. the
            number of items in each partition.
    """

    def __init__(self, partition_size: int):
        self._partition_size = int(partition_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(partition_size={self.partition_size:,})"

    @property
    def partition_size(self) -> int:
        r"""int: The partition size."""
        return self._partition_size

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        return ddp_partitions(items=items, partition_size=self._partition_size)


class SyncParallelPartitioner(BasePartitioner[T]):
    r"""Implements a partitioner that synchronize the partitions in a
    distributed setting.

    This partitioner ensures all the distributed processes have the
    same partitions. The partitions from the main process (rank 0)
    are broadcast to the other processes.

    Args:
        partitioner (``BasePartitioner`` or dict): Specifies a
            partitioner or its configuration.
    """

    def __init__(self, partitioner: Union[BasePartitioner, dict]):
        self._partitioner = setup_partitioner(partitioner)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  partitioner={str_indent(self._partitioner)},\n"
            ")"
        )

    @property
    def partitioner(self) -> BasePartitioner:
        r"""``BasePartitioner``: The partitioner."""
        return self._partitioner

    def partition(
        self, items: Sequence[T], engine: Optional[BaseEngine] = None
    ) -> list[Sequence[T]]:
        partitions = self._partitioner.partition(items, engine)
        # Synchronize the partitions between process so all the distributed processes
        # have the same partitions.
        broadcast_object_list(partitions)
        return partitions
