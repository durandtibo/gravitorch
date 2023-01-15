from unittest.mock import patch

from pytest import mark, raises

from gravitorch.utils.partitioning import (
    ddp_partitions,
    even_partitions,
    fixed_size_partitions,
    select_partition_by_rank,
    split_into_two_partitions,
)

###################################
#     Tests for ddp_partition     #
###################################


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 1)
def test_ddp_partitions_world_size_1_partition_size_1():
    assert ddp_partitions([1, 2, 3, 4], partition_size=1) == [[1]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 1)
def test_ddp_partitions_world_size_1_partition_size_2():
    assert ddp_partitions([1, 2, 3, 4], partition_size=2) == [[1, 2]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 1)
def test_ddp_partitions_world_size_1_partition_size_4():
    assert ddp_partitions([1, 2, 3, 4], partition_size=4) == [[1, 2, 3, 4]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 2)
def test_ddp_partitions_world_size_2_partition_size_1():
    assert ddp_partitions([1, 2, 3, 4], partition_size=1) == [[1], [2]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 2)
def test_ddp_partitions_world_size_2_partition_size_2():
    assert ddp_partitions([1, 2, 3, 4], partition_size=2) == [[1, 2], [3, 4]]


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 1)
def test_ddp_partitions_incorrect_partition_size():
    with raises(ValueError):
        ddp_partitions([1, 2, 3, 4], partition_size=5)


#####################################
#     Tests for even_partitions     #
#####################################


def test_even_partitions_minimum_items():
    assert even_partitions([1, 2], num_partitions=2) == [[1], [2]]


def test_even_partitions_even_drop_remainder_false():
    assert even_partitions([1, 2, 3, 4, 5, 6], num_partitions=3) == [[1, 2], [3, 4], [5, 6]]


def test_even_partitions_even_drop_remainder_true():
    assert even_partitions([1, 2, 3, 4, 5, 6], num_partitions=3, drop_remainder=True) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_even_partitions_not_even_drop_remainder_false():
    assert even_partitions([1, 2, 3, 4, 5, 6, 7, 8], num_partitions=3) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]


def test_even_partitions_not_even_drop_remainder_true():
    assert even_partitions([1, 2, 3, 4, 5, 6, 7, 8], num_partitions=3, drop_remainder=True) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_even_partitions_small_drop_remainder_false():
    assert even_partitions([1, 2], num_partitions=3) == [[1], [2], []]


def test_even_partitions_small_drop_remainder_true():
    assert even_partitions([1, 2], num_partitions=3, drop_remainder=True) == [[], [], []]


def test_even_partitions_type_tuple():
    assert even_partitions((1, 2, 3, 4, 5, 6), num_partitions=3) == [(1, 2), (3, 4), (5, 6)]


###########################################
#     Tests for fixed_size_partitions     #
###########################################


def test_fixed_size_partitions_1():
    assert fixed_size_partitions([1, 2], partition_size=1) == [[1], [2]]


def test_fixed_size_partitions_2():
    assert fixed_size_partitions([1, 2], partition_size=2) == [[1, 2]]


def test_fixed_size_partitions_even_drop_last_false():
    assert fixed_size_partitions([1, 2, 3, 4, 5, 6], partition_size=2) == [[1, 2], [3, 4], [5, 6]]


def test_fixed_size_partitions_even_drop_last_true():
    assert fixed_size_partitions([1, 2, 3, 4, 5, 6], partition_size=2, drop_last=True) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_fixed_size_partitions_not_even_drop_last_false():
    assert fixed_size_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8],
    ]


def test_fixed_size_partitions_not_even_drop_last_true():
    assert fixed_size_partitions([1, 2, 3, 4, 5, 6, 7, 8], partition_size=3, drop_last=True) == [
        [1, 2, 3],
        [4, 5, 6],
    ]


def test_fixed_size_partitions_type_tuple():
    assert fixed_size_partitions((1, 2, 3, 4, 5, 6), partition_size=2) == [(1, 2), (3, 4), (5, 6)]


##############################################
#     Tests for select_partition_by_rank     #
##############################################


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 4)
@patch("gravitorch.utils.partitioning.dist.get_rank", lambda *args, **kwargs: 0)
def test_select_partition_by_rank_rank_0():
    assert select_partition_by_rank([1, 2, 3, 4]) == 1


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 4)
@patch("gravitorch.utils.partitioning.dist.get_rank", lambda *args, **kwargs: 1)
def test_select_partition_by_rank_rank_1():
    assert select_partition_by_rank([1, 2, 3, 4]) == 2


@patch("gravitorch.utils.partitioning.dist.get_world_size", lambda *args, **kwargs: 4)
@patch("gravitorch.utils.partitioning.dist.get_rank", lambda *args, **kwargs: 0)
def test_select_partition_by_rank_incorrect_number_of_partitions():
    with raises(ValueError):
        select_partition_by_rank([1, 2, 3])


###############################################
#     Tests for split_into_two_partitions     #
###############################################


@mark.parametrize(
    "first_partition_ratio,first_len,second_len", ((0.5, 5, 5), (0.3, 3, 7), (0.8, 8, 2))
)
def test_split_into_two_partitions(first_partition_ratio: float, first_len: int, second_len: int):
    first, second = split_into_two_partitions(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        first_partition_ratio=first_partition_ratio,
    )
    assert len(first) == first_len
    assert len(second) == second_len


def test_split_into_two_partitions_same_random_seed():
    items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert split_into_two_partitions(items, random_seed=1) == split_into_two_partitions(
        items, random_seed=1
    )


def test_split_into_two_partitions_different_random_seeds():
    items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert split_into_two_partitions(items, random_seed=1) != split_into_two_partitions(
        items, random_seed=2
    )
