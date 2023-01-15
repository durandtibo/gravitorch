from unittest.mock import Mock, patch

from pytest import mark

from gravitorch.data.partitioners import (
    BasePartitioner,
    DDPPartitioner,
    SyncParallelPartitioner,
)

####################################
#     Tests for DDPPartitioner     #
####################################


def test_ddp_partitioner_str():
    assert str(DDPPartitioner(partition_size=2)).startswith("DDPPartitioner(")


@mark.parametrize("partition_size", (1, 2, 3))
def test_ddp_partitioner_partition_size(partition_size: int):
    assert DDPPartitioner(partition_size=partition_size).partition_size == partition_size


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 2)
def test_ddp_partitioner_partition_partition_size_1():
    assert DDPPartitioner(partition_size=1).partition([1, 2]) == [[1], [2]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 2)
def test_ddp_partitioner_partition_partition_size_2():
    assert DDPPartitioner(partition_size=2).partition([1, 2, 3, 4, 5, 6]) == [[1, 2], [3, 4]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 2)
def test_ddp_partitioner_partition_type_tuple():
    assert DDPPartitioner(partition_size=3).partition((1, 2, 3, 4, 5, 6)) == [(1, 2, 3), (4, 5, 6)]


#############################################
#     Tests for SyncParallelPartitioner     #
#############################################


def test_synch_parallel_partitioner_str():
    assert str(SyncParallelPartitioner(partitioner=Mock(spec=BasePartitioner))).startswith(
        "SyncParallelPartitioner("
    )


def test_synch_parallel_partitioner_partitioner():
    partitioner = Mock(spec=BasePartitioner)
    assert SyncParallelPartitioner(partitioner).partitioner is partitioner


def test_synch_parallel_partitioner_partition():
    partitioner = SyncParallelPartitioner(
        partitioner=Mock(spec=BasePartitioner, partition=Mock(return_value=[[2, 3], [1, 4]]))
    )
    with patch("gravitorch.data.partitioners.distributed.broadcast_object_list") as broadcast_mock:
        assert partitioner.partition([0, 1, 2, 3, 4, 5]) == [[2, 3], [1, 4]]
        broadcast_mock.assert_called_once_with([[2, 3], [1, 4]])
