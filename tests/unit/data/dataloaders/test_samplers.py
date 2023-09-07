from __future__ import annotations

from unittest.mock import patch

import torch
from pytest import fixture, mark, raises
from torch.utils.data import BatchSampler, SequentialSampler

from gravitorch.data.dataloaders.samplers import (
    PartialRandomSampler,
    PartialSequentialSampler,
    ReproducibleBatchSampler,
)

##############################################
#     Tests for ReproducibleBatchSampler     #
##############################################


@fixture
def batch_sampler() -> BatchSampler:
    return BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)


def test_reproducible_batch_sampler_str(batch_sampler: BatchSampler) -> None:
    assert str(ReproducibleBatchSampler(batch_sampler)).startswith("ReproducibleBatchSampler(")


def test_reproducible_batch_sampler_iter(batch_sampler: BatchSampler) -> None:
    assert list(ReproducibleBatchSampler(batch_sampler)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_reproducible_batch_sampler_iter_multiple_sampling(batch_sampler: BatchSampler) -> None:
    assert list(ReproducibleBatchSampler(batch_sampler)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


@mark.parametrize(
    "start_iteration,batch_indices",
    (
        (0, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]),
        (1, [[3, 4, 5], [6, 7, 8], [9]]),
        (2, [[6, 7, 8], [9]]),
        (3, [[9]]),
        (4, []),
    ),
)
def test_reproducible_batch_sampler_iter_start_iteration(
    batch_sampler: BatchSampler, start_iteration: int, batch_indices: list[int]
) -> None:
    rbs = ReproducibleBatchSampler(batch_sampler, start_iteration=start_iteration)
    assert rbs._start_iteration == start_iteration
    assert list(rbs) == batch_indices


def test_reproducible_batch_sampler_iter_incorrect() -> None:
    with raises(
        TypeError, match="Argument batch_sampler should be torch.utils.data.sampler.BatchSampler"
    ):
        ReproducibleBatchSampler([1, 2, 3], start_iteration=-1)


def test_reproducible_batch_sampler_iter_start_iteration_incorrect(
    batch_sampler: BatchSampler,
) -> None:
    with raises(ValueError, match="Argument start_iteration should be positive integer"):
        ReproducibleBatchSampler(batch_sampler, start_iteration=-1)


##############################################
#     Tests for PartialSequentialSampler     #
##############################################


def test_partial_sequential_sampler_str() -> None:
    assert str(PartialSequentialSampler(range(10), num_samples=5)).startswith(
        "PartialSequentialSampler("
    )


def test_partial_sequential_sampler_num_samples_5() -> None:
    assert list(PartialSequentialSampler(range(10), num_samples=5)) == [0, 1, 2, 3, 4]


def test_partial_sequential_sampler_num_samples_10() -> None:
    assert list(PartialSequentialSampler(range(10), num_samples=10)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_partial_sequential_sampler_num_samples_20_but_only_10_examples() -> None:
    assert list(PartialSequentialSampler(range(10), num_samples=20)) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ]


def test_partial_sequential_sampler_incorrect() -> None:
    with raises(ValueError, match="num_samples should be a positive integer value, but got"):
        PartialSequentialSampler(range(10), num_samples=0)


##########################################
#     Tests for PartialRandomSampler     #
##########################################


def test_partial_random_sampler_str() -> None:
    assert str(PartialRandomSampler(range(10), num_samples=5)).startswith("PartialRandomSampler(")


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_5() -> None:
    assert list(PartialRandomSampler(range(10), num_samples=5)) == [6, 5, 4, 0, 8]


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_10() -> None:
    assert list(PartialRandomSampler(range(10), num_samples=10)) == [6, 5, 4, 0, 8, 9, 2, 1, 3, 7]


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_20_but_only_10_examples() -> None:
    assert list(PartialRandomSampler(range(10), num_samples=20)) == [6, 5, 4, 0, 8, 9, 2, 1, 3, 7]


def test_partial_random_sampler_incorrect() -> None:
    with raises(ValueError, match="num_samples should be a positive integer value, but got"):
        PartialRandomSampler(range(10), num_samples=0)
