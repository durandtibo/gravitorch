from pytest import mark

from gravitorch.data.partitioners import EvenPartitioner

#####################################
#     Tests for EvenPartitioner     #
#####################################


def test_fixed_size_partitioner_str():
    assert str(EvenPartitioner(num_partitions=2)).startswith("EvenPartitioner(")


@mark.parametrize("drop_remainder", (True, False))
def test_even_partitioner_drop_remainder(drop_remainder: bool):
    assert (
        EvenPartitioner(num_partitions=2, drop_remainder=drop_remainder).drop_remainder
        == drop_remainder
    )


@mark.parametrize("num_partitions", (1, 2, 3))
def test_even_partitioner_num_partitions(num_partitions: int):
    assert EvenPartitioner(num_partitions=num_partitions).num_partitions == num_partitions


def test_even_partitioner_partition_num_partitions_1():
    assert EvenPartitioner(num_partitions=1).partition([1, 2]) == [[1, 2]]


def test_even_partitioner_partition_num_partitions_2():
    assert EvenPartitioner(num_partitions=2).partition([1, 2]) == [[1], [2]]


def test_even_partitioner_partition_even_drop_remainder_false():
    assert EvenPartitioner(num_partitions=3).partition([1, 2, 3, 4, 5, 6]) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_even_partitioner_partition_even_drop_remainder_true():
    assert EvenPartitioner(num_partitions=3, drop_remainder=True).partition([1, 2, 3, 4, 5, 6]) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_even_partitioner_partition_not_even_drop_remainder_false():
    assert EvenPartitioner(num_partitions=3).partition([1, 2, 3, 4, 5, 6, 7, 8]) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]


def test_even_partitioner_partition_not_even_drop_remainder_true():
    assert EvenPartitioner(num_partitions=3, drop_remainder=True).partition(
        [1, 2, 3, 4, 5, 6, 7, 8]
    ) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_even_partitioner_partition_small_drop_remainder_false():
    assert EvenPartitioner(num_partitions=3).partition([1, 2]) == [[1], [2], []]


def test_even_partitioner_partition_small_drop_remainder_true():
    assert EvenPartitioner(num_partitions=3, drop_remainder=True).partition([1, 2]) == [[], [], []]


def test_even_partitioner_partition_type_tuple():
    assert EvenPartitioner(num_partitions=3).partition((1, 2, 3, 4, 5, 6)) == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]
