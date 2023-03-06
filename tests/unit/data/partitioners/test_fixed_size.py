from pytest import mark

from gravitorch.data.partitioners import FixedSizePartitioner

##########################################
#     Tests for FixedSizePartitioner     #
##########################################


def test_fixed_size_partitioner_str() -> None:
    assert str(FixedSizePartitioner(partition_size=2)).startswith("FixedSizePartitioner(")


@mark.parametrize("drop_last", (True, False))
def test_fixed_size_partitioner_drop_last(drop_last: bool) -> None:
    assert FixedSizePartitioner(partition_size=2, drop_last=drop_last).drop_last == drop_last


@mark.parametrize("partition_size", (1, 2, 3))
def test_fixed_size_partitioner_partition_size(partition_size: int) -> None:
    assert FixedSizePartitioner(partition_size=partition_size).partition_size == partition_size


def test_fixed_size_partitioner_partition_partition_size_1() -> None:
    assert FixedSizePartitioner(partition_size=1).partition([1, 2]) == [[1], [2]]


def test_fixed_size_partitioner_partition_partition_size_2() -> None:
    assert FixedSizePartitioner(partition_size=2).partition([1, 2]) == [[1, 2]]


def test_fixed_size_partitioner_partition_even_drop_last_false() -> None:
    assert FixedSizePartitioner(partition_size=2).partition([1, 2, 3, 4, 5, 6]) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_fixed_size_partitioner_partition_even_drop_last_true() -> None:
    assert FixedSizePartitioner(partition_size=2, drop_last=True).partition([1, 2, 3, 4, 5, 6]) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_fixed_size_partitioner_partition_not_even_drop_last_false() -> None:
    assert FixedSizePartitioner(partition_size=3).partition([1, 2, 3, 4, 5, 6, 7, 8]) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]


def test_fixed_size_partitioner_partition_not_even_drop_last_true() -> None:
    assert FixedSizePartitioner(partition_size=3, drop_last=True).partition(
        [1, 2, 3, 4, 5, 6, 7, 8]
    ) == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_fixed_size_partitioner_partition_type_tuple() -> None:
    assert FixedSizePartitioner(partition_size=2).partition((1, 2, 3, 4, 5, 6)) == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]
