__all__ = [
    "ddp_partitions",
    "even_partitions",
    "fixed_size_partitions",
    "select_partition_by_rank",
    "split_into_two_partitions",
]

import math
from collections.abc import Sequence
from typing import TypeVar

import torch

from gravitorch import distributed as dist
from gravitorch.utils.seed import get_torch_generator

T = TypeVar("T")


def ddp_partitions(items: Sequence[T], partition_size: int) -> list[Sequence[T]]:
    r"""Creates non-overlapping partitions of a sequence of items in a
    Distributed Data Parallel (DDP) setting.

    All the partitions have the same number of items. The number of
    partitions is the world size of the distributed system.

    Args:
        items (sequence): Specifies the sequence to partition. The
            length should be greater or equal to
            ``partition_size * world_size``.
        partition_size (int): Specifies the partition size i.e. the
            number of items in each partition.

    Returns:
        list: A partition of the items.

    Raises:
        ValueError if ``items`` has less than `
            `partition_size * world_size`` items.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.partitioning import ddp_partitions
        # World size 1
        >>> ddp_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=2)
        [[1, 2]]
        >>> ddp_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3)
        [[1, 2, 3]]
        # World size 2
        >>> ddp_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=2)
        [[1, 2], [3, 4]]
        >>> ddp_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3)
        [[1, 2, 3], [4, 5, 6]]
    """
    world_size = dist.get_world_size()
    if len(items) < partition_size * world_size:
        raise ValueError(
            f"Incorrect `partition_size` {partition_size:,}. The partition size should be lower "
            f"than {len(items) // world_size:,}"
        )
    return [items[i * partition_size : (i + 1) * partition_size] for i in range(world_size)]


def even_partitions(
    items: Sequence[T], num_partitions: int, drop_remainder: bool = False
) -> list[Sequence[T]]:
    r"""Creates non-overlapping partitions of a sequence of items.

    The items are distributed evenly across partitions, starting by
    the first.

    Args:
        items (sequence): Specifies the sequence to partition.
        num_partitions (int): Specifies the number of partitions.
        drop_remainder (bool, optional): If ``True``, it drops the
            last items if the number of items is not evenly divisible
            by ``num_partitions``.

    Returns:
        list: A partition of the items.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.partitioning import even_partitions
        >>> even_partitions([1, 2, 3, 4, 5, 6, 7, 8], num_partitions=3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]
        >>> even_partitions([1, 2, 3, 4, 5, 6, 7, 8], num_partitions=3, drop_remainder=True)
        [[1, 2], [3, 4], [5, 6]]
    """
    rounding_op = math.floor if drop_remainder else math.ceil
    partition_size = rounding_op(len(items) / num_partitions)
    return [items[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)]


def fixed_size_partitions(
    items: Sequence[T], partition_size: int, drop_last: bool = False
) -> list[Sequence[T]]:
    r"""Creates non-overlapping partitions of a sequence of items.

    The items are distributed across partitions, starting by the first.

    Args:
        items (sequence): Specifies the sequence to partition.
        partition_size (int): Specifies the number of items per
            partitions.
        drop_last (bool, optional): If ``True``, it drops the last
            items if the number of items is not evenly divisible by
            ``partition_size``.

    Returns:
        list: A partition of the items.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.partitioning import fixed_size_partitions
        >>> fixed_size_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3)
        [[1, 2, 3], [4, 5, 6], [7, 8]]
        >>> fixed_size_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3, drop_last=True)
        [[1, 2, 3], [4, 5, 6]]
    """
    rounding_op = math.floor if drop_last else math.ceil
    num_partitions = rounding_op(len(items) / partition_size)
    return [items[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)]


def select_partition_by_rank(partitions: Sequence[T]) -> T:
    r"""Selects a partition by using the current distributed rank.

    Args:
        partitions: Specifies the sequence of partitions

    Returns:
        The selected partition for the current distributed rank.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.partitioning import select_partition_by_rank
        # Rank 0
        >>> select_partition_by_rank([1, 2, 3, 4])
        1
        # Rank 1
        >>> select_partition_by_rank([1, 2, 3, 4])
        2
    """
    world_size = dist.get_world_size()
    if len(partitions) < world_size:
        raise ValueError(
            f"Incorrect number of partitions. The number of partitions ({len(partitions)}) "
            f"has to be greater or equal to the distributed world size ({world_size})"
        )
    return partitions[dist.get_rank()]


def split_into_two_partitions(
    items: Sequence[T], first_partition_ratio: float = 0.8, random_seed: int = 4979957926458772275
) -> tuple[tuple[T, ...], tuple[T, ...]]:
    r"""Splits the items into two partitions.

    This function can be used to create train/val or train/test
    dataset splits.

    Args:
        items (sequence): Specifies the items to split into two
            partitions.
        first_partition_ratio (float, optional): Specifies the ratio
            of the first partition. ``0.8`` means that 80% of the
            items will be split into the first partition.
        random_seed (int, optional): Specifies the random seed used
            to create the partitions. Default: ``4979957926458772275``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.partitioning import split_into_two_partitions
        >>> first, second = split_into_two_partitions([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> first
        (2, 7, 4, 1, 6, 5, 3, 8)
        >>> second
        (9, 0)
    """
    permutation = torch.randperm(len(items), generator=get_torch_generator(random_seed))
    num_items_first_partition = int(first_partition_ratio * len(items))
    return (
        tuple(items[i] for i in permutation[:num_items_first_partition]),
        tuple(items[i] for i in permutation[num_items_first_partition:]),
    )
