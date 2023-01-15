from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark

from gravitorch import constants as ct
from gravitorch.data.datapipes.iter import (
    DictBatcherSrcIterDataPipe,
    TupleBatcherSrcIterDataPipe,
)

################################################
#     Tests for DictBatcherSrcIterDataPipe     #
################################################


def test_dict_batcher_src_iter_datapipe_str():
    assert str(
        DictBatcherSrcIterDataPipe(
            {ct.INPUT: torch.zeros(10, 2), ct.TARGET: torch.zeros(10)}, batch_size=32
        )
    ).startswith("DictBatcherSrcIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_dict_dict_shuffler_iter_datapipe_iter_random_seed(random_seed: int):
    assert (
        DictBatcherSrcIterDataPipe({}, batch_size=32, random_seed=random_seed).random_seed
        == random_seed
    )


def test_dict_batcher_src_iter_datapipe_batch_size_2():
    datapipe = DictBatcherSrcIterDataPipe(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=2,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[0, 1], [2, 3]]))
    assert batch[ct.TARGET].equal(torch.tensor([[0], [1]]))
    assert len(tuple(datapipe)) == 5


def test_dict_batcher_src_iter_datapipe_batch_size_4():
    datapipe = DictBatcherSrcIterDataPipe(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=4,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]))
    assert batch[ct.TARGET].equal(torch.tensor([[0], [1], [2], [3]]))
    assert len(tuple(datapipe)) == 3


@patch(
    "gravitorch.data.datapipes.iter.source_batch.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_dict_batcher_src_iter_datapipe_shuffle_true():
    datapipe = DictBatcherSrcIterDataPipe(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]))
    assert batch[ct.TARGET].equal(torch.tensor([[5], [4], [6], [3]]))
    assert len(tuple(datapipe)) == 3


def test_dict_batcher_src_iter_datapipe_generator_same_random_seed():
    assert objects_are_equal(
        tuple(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_dict_batcher_src_iter_datapipe_generator_different_random_seeds():
    assert not objects_are_equal(
        tuple(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_dict_batcher_src_iter_datapipe_generator_repeat():
    datapipe = DictBatcherSrcIterDataPipe(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_dict_batcher_src_iter_datapipe_len_batch_size_2():
    assert (
        len(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=2
            )
        )
        == 5
    )


def test_dict_batcher_src_iter_datapipe_len_batch_size_4():
    assert (
        len(
            DictBatcherSrcIterDataPipe(
                data={ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=4
            )
        )
        == 3
    )


#################################################
#     Tests for TupleBatcherSrcIterDataPipe     #
#################################################


def test_tuple_batcher_src_iter_datapipe_str():
    assert str(
        TupleBatcherSrcIterDataPipe((torch.zeros(10, 2), torch.zeros(10)), batch_size=32)
    ).startswith("TupleBatcherSrcIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_tensor_dict_shuffler_iter_datapipe_iter_random_seed(random_seed: int):
    assert (
        TupleBatcherSrcIterDataPipe([], batch_size=32, random_seed=random_seed).random_seed
        == random_seed
    )


def test_tuple_batcher_src_iter_datapipe_batch_size_2():
    datapipe = TupleBatcherSrcIterDataPipe(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=2,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[0, 1], [2, 3]]))
    assert batch[1].equal(torch.tensor([[0], [1]]))
    assert len(tuple(datapipe)) == 5


def test_tuple_batcher_src_iter_datapipe_batch_size_4():
    datapipe = TupleBatcherSrcIterDataPipe(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]))
    assert batch[1].equal(torch.tensor([[0], [1], [2], [3]]))
    assert len(tuple(datapipe)) == 3


@patch(
    "gravitorch.data.datapipes.iter.source_batch.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_tuple_batcher_src_iter_datapipe_shuffle_true():
    datapipe = TupleBatcherSrcIterDataPipe(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]))
    assert batch[1].equal(torch.tensor([[5], [4], [6], [3]]))
    assert len(tuple(datapipe)) == 3


def test_tuple_batcher_src_iter_datapipe_generator_same_random_seed():
    assert objects_are_equal(
        tuple(
            TupleBatcherSrcIterDataPipe(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcherSrcIterDataPipe(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_tuple_batcher_src_iter_datapipe_generator_different_random_seeds():
    assert not objects_are_equal(
        tuple(
            TupleBatcherSrcIterDataPipe(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcherSrcIterDataPipe(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_tuple_batcher_src_iter_datapipe_generator_repeat():
    datapipe = TupleBatcherSrcIterDataPipe(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_tuple_batcher_src_iter_datapipe_len_batch_size_2():
    assert len(TupleBatcherSrcIterDataPipe([torch.ones(10), torch.zeros(10)], batch_size=2)) == 5


def test_tuple_batcher_src_iter_datapipe_len_batch_size_4():
    assert len(TupleBatcherSrcIterDataPipe([torch.ones(10), torch.zeros(10)], batch_size=4)) == 3
