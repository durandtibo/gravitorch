from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch.data.partitioners import (
    BasePartitioner,
    EpochShufflePartitioner,
    FixedSizePartitioner,
)
from gravitorch.engines import BaseEngine

#############################################
#     Tests for EpochShufflePartitioner     #
#############################################


def test_epoch_shuffle_partitioner_str() -> None:
    assert str(EpochShufflePartitioner(Mock(spec=BasePartitioner))).startswith(
        "EpochShufflePartitioner("
    )


def test_epoch_shuffle_partitioner_partitioner() -> None:
    partitioner = FixedSizePartitioner(partition_size=4)
    assert EpochShufflePartitioner(partitioner).partitioner is partitioner


def test_epoch_shuffle_partitioner_dict() -> None:
    assert isinstance(
        EpochShufflePartitioner(
            {
                OBJECT_TARGET: "gravitorch.data.partitioners.FixedSizePartitioner",
                "partition_size": 4,
            }
        ).partitioner,
        FixedSizePartitioner,
    )


@mark.parametrize("random_seed", (1, 2))
def test_epoch_shuffle_partitioner_random_seed(random_seed: int):
    assert (
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=random_seed
        ).random_seed
        == random_seed
    )


@patch(
    "gravitorch.data.partitioners.shuffling.torch.randperm",
    lambda *args, **kwargs: torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
)
def test_epoch_shuffle_partitioner_partition() -> None:
    partitioner = EpochShufflePartitioner(FixedSizePartitioner(partition_size=4))
    assert partitioner.partition(list(range(10))) == [[9, 8, 7, 6], [5, 4, 3, 2], [1, 0]]


def test_epoch_shuffle_partitioner_partition_same_random_seed() -> None:
    items = list(range(10))
    assert objects_are_equal(
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(items),
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(items),
    )


def test_epoch_shuffle_partitioner_partition_different_random_seeds() -> None:
    items = list(range(10))
    assert not objects_are_equal(
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=7984733130308401219
        ).partition(items),
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=91344953463927045
        ).partition(items),
    )


def test_epoch_shuffle_partitioner_partition_same_random_seed_and_epoch() -> None:
    engine = Mock(spec=BaseEngine, epoch=42)
    items = list(range(10))
    assert objects_are_equal(
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(items, engine),
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(items, engine),
    )


def test_epoch_shuffle_partitioner_partition_same_random_seed_and_different_epochs() -> None:
    items = list(range(10))
    assert not objects_are_equal(
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(
            items, engine=Mock(spec=BaseEngine, epoch=42)
        ),
        EpochShufflePartitioner(FixedSizePartitioner(partition_size=4)).partition(
            items, engine=Mock(spec=BaseEngine, epoch=35)
        ),
    )


def test_epoch_shuffle_partitioner_partition_different_random_seeds_and_same_epoch() -> None:
    engine = Mock(spec=BaseEngine, epoch=42)
    items = list(range(10))
    assert not objects_are_equal(
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=7984733130308401219
        ).partition(items, engine),
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=91344953463927045
        ).partition(items, engine),
    )


def test_epoch_shuffle_partitioner_partition_different_random_seeds_and_epochs() -> None:
    items = list(range(10))
    assert not objects_are_equal(
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=7984733130308401219
        ).partition(items, engine=Mock(spec=BaseEngine, epoch=42)),
        EpochShufflePartitioner(
            FixedSizePartitioner(partition_size=4), random_seed=91344953463927045
        ).partition(items, engine=Mock(spec=BaseEngine, epoch=35)),
    )
