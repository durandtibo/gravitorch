from unittest.mock import Mock, patch

from gravitorch.data.partitioners import SequentialPartitioner
from gravitorch.engines import BaseEngine

###########################################
#     Tests for SequentialPartitioner     #
###########################################


def test_sequential_partitioner_str() -> None:
    assert str(SequentialPartitioner()) == "SequentialPartitioner(partition_size=1)"


def test_sequential_partitioner_partition_partition_size_1_no_epoch() -> None:
    assert SequentialPartitioner().partition([1, 2]) == [[1]]


def test_sequential_partitioner_partition_partition_size_1_epoch_1() -> None:
    assert SequentialPartitioner().partition([1, 2], engine=Mock(spec=BaseEngine, epoch=1)) == [[2]]


def test_sequential_partitioner_partition_partition_size_1_epoch_2() -> None:
    assert SequentialPartitioner().partition([1, 2], engine=Mock(spec=BaseEngine, epoch=2)) == [[1]]


def test_sequential_partitioner_partition_partition_size_2_no_epoch() -> None:
    assert SequentialPartitioner(partition_size=2).partition([1, 2, 3]) == [[1, 2]]


def test_sequential_partitioner_partition_partition_size_2_epoch_1() -> None:
    assert SequentialPartitioner(partition_size=2).partition(
        [1, 2, 3], engine=Mock(spec=BaseEngine, epoch=1)
    ) == [[3, 1]]


def test_sequential_partitioner_partition_partition_size_2_epoch_2() -> None:
    assert SequentialPartitioner(partition_size=2).partition(
        [1, 2, 3], engine=Mock(spec=BaseEngine, epoch=2)
    ) == [[2, 3]]


def test_sequential_partitioner_partition_partition_size_3_no_epoch() -> None:
    assert SequentialPartitioner(partition_size=3).partition([1, 2]) == [[1, 2, 1]]


def test_sequential_partitioner_partition_partition_size_3_epoch_1() -> None:
    assert SequentialPartitioner(partition_size=3).partition(
        [1, 2], engine=Mock(spec=BaseEngine, epoch=1)
    ) == [[2, 1, 2]]


@patch("gravitorch.data.partitioners.sequential.dist.get_world_size", lambda *args, **kwargs: 3)
def test_sequential_partitioner_partition_world_size_3_no_epoch() -> None:
    assert SequentialPartitioner().partition([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) == [[0], [1], [2]]


@patch("gravitorch.data.partitioners.sequential.dist.get_world_size", lambda *args, **kwargs: 3)
def test_sequential_partitioner_partition_world_size_3_epoch_1() -> None:
    assert SequentialPartitioner().partition(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], engine=Mock(spec=BaseEngine, epoch=1)
    ) == [[3], [4], [5]]


@patch("gravitorch.data.partitioners.sequential.dist.get_world_size", lambda *args, **kwargs: 3)
def test_sequential_partitioner_partition_world_size_3_epoch_2() -> None:
    assert SequentialPartitioner().partition(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], engine=Mock(spec=BaseEngine, epoch=2)
    ) == [[6], [7], [8]]


@patch("gravitorch.data.partitioners.sequential.dist.get_world_size", lambda *args, **kwargs: 3)
def test_sequential_partitioner_partition_world_size_3_epoch_3() -> None:
    assert SequentialPartitioner().partition(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], engine=Mock(spec=BaseEngine, epoch=3)
    ) == [[9], [0], [1]]


def test_sequential_partitioner_partition_empty() -> None:
    assert SequentialPartitioner().partition([]) == [[]]
