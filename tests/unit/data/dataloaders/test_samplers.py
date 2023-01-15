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
def batch_sampler():
    return BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)


def test_reproducible_batch_sampler(batch_sampler):
    rbs = ReproducibleBatchSampler(batch_sampler)
    assert list(rbs) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_reproducible_batch_sampler_multiple_sampling(batch_sampler):
    rbs = ReproducibleBatchSampler(batch_sampler)
    assert list(rbs) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert list(rbs) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


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
def test_reproducible_batch_sampler_start_iteration(batch_sampler, start_iteration, batch_indices):
    rbs = ReproducibleBatchSampler(batch_sampler, start_iteration=start_iteration)
    assert rbs._start_iteration == start_iteration
    assert list(rbs) == batch_indices


def test_reproducible_batch_sampler_incorrect():
    with raises(TypeError):
        ReproducibleBatchSampler([1, 2, 3], start_iteration=-1)


def test_reproducible_batch_sampler_start_iteration_incorrect(batch_sampler):
    with raises(ValueError):
        ReproducibleBatchSampler(batch_sampler, start_iteration=-1)


##############################################
#     Tests for PartialSequentialSampler     #
##############################################


def test_partial_sequential_sampler_num_samples_5():
    sampler = PartialSequentialSampler(range(10), num_samples=5)
    assert list(sampler) == [0, 1, 2, 3, 4]


def test_partial_sequential_sampler_num_samples_10():
    sampler = PartialSequentialSampler(range(10), num_samples=10)
    assert list(sampler) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_partial_sequential_sampler_num_samples_20_but_only_10_examples():
    sampler = PartialSequentialSampler(range(10), num_samples=20)
    assert list(sampler) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_partial_sequential_sampler_incorrect():
    with raises(ValueError):
        PartialSequentialSampler(range(10), num_samples=0)


##########################################
#     Tests for PartialRandomSampler     #
##########################################


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_5():
    sampler = PartialRandomSampler(range(10), num_samples=5)
    assert list(sampler) == [6, 5, 4, 0, 8]


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_10():
    sampler = PartialRandomSampler(range(10), num_samples=10)
    assert list(sampler) == [6, 5, 4, 0, 8, 9, 2, 1, 3, 7]


@patch("torch.randperm", lambda x: torch.tensor([6, 5, 4, 0, 8, 9, 2, 1, 3, 7]))
def test_partial_random_sampler_num_samples_20_but_only_10_examples():
    sampler = PartialRandomSampler(range(10), num_samples=20)
    assert list(sampler) == [6, 5, 4, 0, 8, 9, 2, 1, 3, 7]


def test_partial_random_sampler_incorrect():
    with raises(ValueError):
        PartialRandomSampler(range(10), num_samples=0)
