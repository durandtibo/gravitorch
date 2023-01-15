from gravitorch.data.partitioners import TrivialPartitioner

########################################
#     Tests for TrivialPartitioner     #
########################################


def test_fixed_size_partitioner_str():
    assert str(TrivialPartitioner()) == "TrivialPartitioner()"


def test_fixed_size_partitioner_partition():
    assert TrivialPartitioner().partition([1, 2]) == [[1, 2]]


def test_fixed_size_partitioner_partition_empty():
    assert TrivialPartitioner().partition([]) == [[]]
