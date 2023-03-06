from gravitorch.data.partitioners import TrivialPartitioner

########################################
#     Tests for TrivialPartitioner     #
########################################


def test_fixed_size_partitioner_str() -> None:
    assert str(TrivialPartitioner()) == "TrivialPartitioner()"


def test_fixed_size_partitioner_partition() -> None:
    assert TrivialPartitioner().partition([1, 2]) == [[1, 2]]


def test_fixed_size_partitioner_partition_empty() -> None:
    assert TrivialPartitioner().partition([]) == [[]]
