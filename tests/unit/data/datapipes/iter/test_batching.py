from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark

from gravitorch import constants as ct
from gravitorch.data.datapipes.iter import DictBatcher, TupleBatcher

#################################
#     Tests for DictBatcher     #
#################################


def test_dict_batcher_str():
    assert str(
        DictBatcher({ct.INPUT: torch.zeros(10, 2), ct.TARGET: torch.zeros(10)}, batch_size=32)
    ).startswith("DictBatcherIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_dict_dict_shuffler_iter_datapipe_iter_random_seed(random_seed: int):
    assert DictBatcher({}, batch_size=32, random_seed=random_seed).random_seed == random_seed


def test_dict_batcher_batch_size_2():
    datapipe = DictBatcher(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=2,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[0, 1], [2, 3]]))
    assert batch[ct.TARGET].equal(torch.tensor([[0], [1]]))
    assert len(tuple(datapipe)) == 5


def test_dict_batcher_batch_size_4():
    datapipe = DictBatcher(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=4,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]))
    assert batch[ct.TARGET].equal(torch.tensor([[0], [1], [2], [3]]))
    assert len(tuple(datapipe)) == 3


@patch(
    "gravitorch.data.datapipes.iter.batching.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_dict_batcher_shuffle_true():
    datapipe = DictBatcher(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[ct.INPUT].equal(torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]))
    assert batch[ct.TARGET].equal(torch.tensor([[5], [4], [6], [3]]))
    assert len(tuple(datapipe)) == 3


def test_dict_batcher_generator_same_random_seed():
    assert objects_are_equal(
        tuple(
            DictBatcher(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcher(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_dict_batcher_generator_different_random_seeds():
    assert not objects_are_equal(
        tuple(
            DictBatcher(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcher(
                data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_dict_batcher_generator_repeat():
    datapipe = DictBatcher(
        data={ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_dict_batcher_len_batch_size_2():
    assert (
        len(
            DictBatcher(
                data={ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=2
            )
        )
        == 5
    )


def test_dict_batcher_len_batch_size_4():
    assert (
        len(
            DictBatcher(
                data={ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=4
            )
        )
        == 3
    )


##################################
#     Tests for TupleBatcher     #
##################################


def test_tuple_batcher_str():
    assert str(TupleBatcher((torch.zeros(10, 2), torch.zeros(10)), batch_size=32)).startswith(
        "TupleBatcherIterDataPipe("
    )


@mark.parametrize("random_seed", (1, 2))
def test_tensor_dict_shuffler_iter_datapipe_iter_random_seed(random_seed: int):
    assert TupleBatcher([], batch_size=32, random_seed=random_seed).random_seed == random_seed


def test_tuple_batcher_batch_size_2():
    datapipe = TupleBatcher(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=2,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[0, 1], [2, 3]]))
    assert batch[1].equal(torch.tensor([[0], [1]]))
    assert len(tuple(datapipe)) == 5


def test_tuple_batcher_batch_size_4():
    datapipe = TupleBatcher(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]))
    assert batch[1].equal(torch.tensor([[0], [1], [2], [3]]))
    assert len(tuple(datapipe)) == 3


@patch(
    "gravitorch.data.datapipes.iter.batching.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_tuple_batcher_shuffle_true():
    datapipe = TupleBatcher(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
        shuffle=True,
    )
    batch = next(iter(datapipe))
    assert len(batch) == 2
    assert batch[0].equal(torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]))
    assert batch[1].equal(torch.tensor([[5], [4], [6], [3]]))
    assert len(tuple(datapipe)) == 3


def test_tuple_batcher_generator_same_random_seed():
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcher(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_tuple_batcher_generator_different_random_seeds():
    assert not objects_are_equal(
        tuple(
            TupleBatcher(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcher(
                tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_tuple_batcher_generator_repeat():
    datapipe = TupleBatcher(
        tensors=[torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_tuple_batcher_len_batch_size_2():
    assert len(TupleBatcher([torch.ones(10), torch.zeros(10)], batch_size=2)) == 5


def test_tuple_batcher_len_batch_size_4():
    assert len(TupleBatcher([torch.ones(10), torch.zeros(10)], batch_size=4)) == 3
