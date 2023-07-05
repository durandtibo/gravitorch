from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor
from torch.utils.data.datapipes.iter import IterableWrapper

from gravitorch import constants as ct
from gravitorch.datapipes.iter import DictBatcher, SourceWrapper, TupleBatcher
from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2

#################################
#     Tests for DictBatcher     #
#################################


def test_dict_batcher_str() -> None:
    assert str(
        DictBatcher({ct.INPUT: torch.zeros(10, 2), ct.TARGET: torch.zeros(10)}, batch_size=32)
    ).startswith("DictBatcherIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_dict_batcher_iter_random_seed(random_seed: int) -> None:
    assert DictBatcher({}, batch_size=32, random_seed=random_seed).random_seed == random_seed


def test_dict_batcher_iter_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
                batch_size=4,
            )
        ),
        (
            {
                ct.INPUT: torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
                ct.TARGET: torch.tensor([[0], [1], [2], [3]]),
            },
            {
                ct.INPUT: torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]]),
                ct.TARGET: torch.tensor([[4], [5], [6], [7]]),
            },
            {ct.INPUT: torch.tensor([[16, 17], [18, 19]]), ct.TARGET: torch.tensor([[8], [9]])},
        ),
    )


def test_dict_batcher_iter_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
                batch_size=2,
            )
        ),
        (
            {ct.INPUT: torch.tensor([[0, 1], [2, 3]]), ct.TARGET: torch.tensor([[0], [1]])},
            {ct.INPUT: torch.tensor([[4, 5], [6, 7]]), ct.TARGET: torch.tensor([[2], [3]])},
            {ct.INPUT: torch.tensor([[8, 9], [10, 11]]), ct.TARGET: torch.tensor([[4], [5]])},
            {ct.INPUT: torch.tensor([[12, 13], [14, 15]]), ct.TARGET: torch.tensor([[6], [7]])},
            {ct.INPUT: torch.tensor([[16, 17], [18, 19]]), ct.TARGET: torch.tensor([[8], [9]])},
        ),
    )


def test_dict_batcher_iter_datapipe_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            DictBatcher(
                SourceWrapper(
                    (
                        {
                            ct.INPUT: torch.arange(20).view(10, 2),
                            ct.TARGET: torch.arange(10).view(10, 1),
                        },
                        {
                            ct.INPUT: -torch.arange(20).view(10, 2),
                            ct.TARGET: torch.arange(10).view(10, 1),
                        },
                    )
                ),
                batch_size=4,
            )
        ),
        (
            {
                ct.INPUT: torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]),
                ct.TARGET: torch.tensor([[0], [1], [2], [3]]),
            },
            {
                ct.INPUT: torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]]),
                ct.TARGET: torch.tensor([[4], [5], [6], [7]]),
            },
            {ct.INPUT: torch.tensor([[16, 17], [18, 19]]), ct.TARGET: torch.tensor([[8], [9]])},
            {
                ct.INPUT: torch.tensor([[-0, -1], [-2, -3], [-4, -5], [-6, -7]]),
                ct.TARGET: torch.tensor([[0], [1], [2], [3]]),
            },
            {
                ct.INPUT: torch.tensor([[-8, -9], [-10, -11], [-12, -13], [-14, -15]]),
                ct.TARGET: torch.tensor([[4], [5], [6], [7]]),
            },
            {ct.INPUT: torch.tensor([[-16, -17], [-18, -19]]), ct.TARGET: torch.tensor([[8], [9]])},
        ),
    )


@patch(
    "gravitorch.datapipes.iter.batching.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_dict_batcher_iter_shuffle_true() -> None:
    assert objects_are_equal(
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10).view(10, 1)},
                batch_size=4,
                shuffle=True,
            )
        ),
        (
            {
                ct.INPUT: torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]),
                ct.TARGET: torch.tensor([[5], [4], [6], [3]]),
            },
            {
                ct.INPUT: torch.tensor([[14, 15], [4, 5], [16, 17], [2, 3]]),
                ct.TARGET: torch.tensor([[7], [2], [8], [1]]),
            },
            {ct.INPUT: torch.tensor([[18, 19], [0, 1]]), ct.TARGET: torch.tensor([[9], [0]])},
        ),
    )


def test_dict_batcher_generator_same_random_seed() -> None:
    assert objects_are_equal(
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_dict_batcher_generator_different_random_seeds() -> None:
    assert not objects_are_equal(
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            DictBatcher(
                {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_dict_batcher_generator_repeat() -> None:
    datapipe = DictBatcher(
        {ct.INPUT: torch.arange(20).view(10, 2), ct.TARGET: torch.arange(10)},
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_dict_batcher_len_datapipe() -> None:
    with raises(TypeError, match="DictBatcherIterDataPipe instance doesn't have valid length"):
        len(DictBatcher(SourceWrapper({}), batch_size=4))


def test_dict_batcher_len_batch_size_2() -> None:
    assert (
        len(DictBatcher({ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=2))
        == 5
    )


def test_dict_batcher_len_batch_size_4() -> None:
    assert (
        len(DictBatcher({ct.INPUT: torch.ones(10, 5), ct.TARGET: torch.zeros(10)}, batch_size=4))
        == 3
    )


def test_dict_batcher_getstate() -> None:
    state = DictBatcher(
        {
            ct.INPUT: torch.arange(20).view(10, 2),
            ct.TARGET: torch.arange(10).view(10, 1),
        },
        batch_size=2,
    ).__getstate__()
    assert len(state) == 4
    assert objects_are_equal(
        state["_datapipe_or_data"],
        {
            ct.INPUT: torch.arange(20).view(10, 2),
            ct.TARGET: torch.arange(10).view(10, 1),
        },
    )
    assert state["_batch_size"] == 2
    assert not state["_shuffle"]
    assert isinstance(state["_generator"], Tensor)


def test_dict_batcher_setstate() -> None:
    state = torch.Generator().get_state()
    dp = DictBatcher({}, batch_size=2)
    dp.__setstate__(
        {
            "_datapipe_or_data": {
                ct.INPUT: torch.arange(20).view(10, 2),
                ct.TARGET: torch.arange(10).view(10, 1),
            },
            "_batch_size": 4,
            "_shuffle": True,
            "_generator": state,
        }
    )
    assert objects_are_equal(
        dp._datapipe_or_data,
        {
            ct.INPUT: torch.arange(20).view(10, 2),
            ct.TARGET: torch.arange(10).view(10, 1),
        },
    )
    assert dp._batch_size == 4
    assert dp._shuffle
    assert dp._generator.get_state().equal(state)


def test_dict_batcher_state_repeat() -> None:
    dp = DictBatcher(
        {
            ct.INPUT: torch.arange(40).view(20, 2),
            ct.TARGET: torch.arange(20).view(20, 1),
        },
        batch_size=4,
        shuffle=True,
    )
    state = dp.__getstate__()
    batches1 = tuple(dp)
    batches2 = tuple(dp)
    dp.__setstate__(state)
    batches3 = tuple(dp)
    assert not objects_are_equal(batches1, batches2)
    assert objects_are_equal(batches1, batches3)


@torchdata_available
def test_dict_batcher_dataloader2_data() -> None:
    assert objects_are_equal(
        tuple(
            DataLoader2(
                DictBatcher(
                    {
                        ct.INPUT: torch.arange(20).view(10, 2),
                        ct.TARGET: torch.arange(10).view(10, 1),
                    },
                    batch_size=2,
                )
            )
        ),
        (
            {ct.INPUT: torch.tensor([[0, 1], [2, 3]]), ct.TARGET: torch.tensor([[0], [1]])},
            {ct.INPUT: torch.tensor([[4, 5], [6, 7]]), ct.TARGET: torch.tensor([[2], [3]])},
            {ct.INPUT: torch.tensor([[8, 9], [10, 11]]), ct.TARGET: torch.tensor([[4], [5]])},
            {ct.INPUT: torch.tensor([[12, 13], [14, 15]]), ct.TARGET: torch.tensor([[6], [7]])},
            {ct.INPUT: torch.tensor([[16, 17], [18, 19]]), ct.TARGET: torch.tensor([[8], [9]])},
        ),
    )


@torchdata_available
def test_dict_batcher_dataloader2_datapipe() -> None:
    assert objects_are_equal(
        tuple(
            DataLoader2(
                DictBatcher(
                    IterableWrapper(
                        [
                            {
                                ct.INPUT: torch.arange(20).view(10, 2),
                                ct.TARGET: torch.arange(10).view(10, 1),
                            },
                            {
                                ct.INPUT: torch.arange(10).view(5, 2),
                                ct.TARGET: torch.arange(5).view(5, 1),
                            },
                        ]
                    ),
                    batch_size=2,
                )
            )
        ),
        (
            {ct.INPUT: torch.tensor([[0, 1], [2, 3]]), ct.TARGET: torch.tensor([[0], [1]])},
            {ct.INPUT: torch.tensor([[4, 5], [6, 7]]), ct.TARGET: torch.tensor([[2], [3]])},
            {ct.INPUT: torch.tensor([[8, 9], [10, 11]]), ct.TARGET: torch.tensor([[4], [5]])},
            {ct.INPUT: torch.tensor([[12, 13], [14, 15]]), ct.TARGET: torch.tensor([[6], [7]])},
            {ct.INPUT: torch.tensor([[16, 17], [18, 19]]), ct.TARGET: torch.tensor([[8], [9]])},
            {ct.INPUT: torch.tensor([[0, 1], [2, 3]]), ct.TARGET: torch.tensor([[0], [1]])},
            {ct.INPUT: torch.tensor([[4, 5], [6, 7]]), ct.TARGET: torch.tensor([[2], [3]])},
            {ct.INPUT: torch.tensor([[8, 9]]), ct.TARGET: torch.tensor([[4]])},
        ),
    )


##################################
#     Tests for TupleBatcher     #
##################################


def test_tuple_batcher_str() -> None:
    assert str(TupleBatcher((torch.zeros(10, 2), torch.zeros(10)), batch_size=32)).startswith(
        "TupleBatcherIterDataPipe("
    )


@mark.parametrize("random_seed", (1, 2))
def test_tuple_batcher_random_seed(random_seed: int) -> None:
    assert TupleBatcher([], batch_size=32, random_seed=random_seed).random_seed == random_seed


def test_tuple_batcher_iter_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
            )
        ),
        (
            (torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]), torch.tensor([[0], [1], [2], [3]])),
            (
                torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]]),
                torch.tensor([[4], [5], [6], [7]]),
            ),
            (torch.tensor([[16, 17], [18, 19]]), torch.tensor([[8], [9]])),
        ),
    )


def test_tuple_batcher_iter_batch_size_2() -> None:
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=2,
            )
        ),
        (
            (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0], [1]])),
            (torch.tensor([[4, 5], [6, 7]]), torch.tensor([[2], [3]])),
            (torch.tensor([[8, 9], [10, 11]]), torch.tensor([[4], [5]])),
            (torch.tensor([[12, 13], [14, 15]]), torch.tensor([[6], [7]])),
            (torch.tensor([[16, 17], [18, 19]]), torch.tensor([[8], [9]])),
        ),
    )


def test_tuple_batcher_iter_datapipe_batch_size_4() -> None:
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                SourceWrapper(
                    (
                        [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                        [-torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                    )
                ),
                batch_size=4,
            )
        ),
        (
            (torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]), torch.tensor([[0], [1], [2], [3]])),
            (
                torch.tensor([[8, 9], [10, 11], [12, 13], [14, 15]]),
                torch.tensor([[4], [5], [6], [7]]),
            ),
            (torch.tensor([[16, 17], [18, 19]]), torch.tensor([[8], [9]])),
            (
                torch.tensor([[-0, -1], [-2, -3], [-4, -5], [-6, -7]]),
                torch.tensor([[0], [1], [2], [3]]),
            ),
            (
                torch.tensor([[-8, -9], [-10, -11], [-12, -13], [-14, -15]]),
                torch.tensor([[4], [5], [6], [7]]),
            ),
            (torch.tensor([[-16, -17], [-18, -19]]), torch.tensor([[8], [9]])),
        ),
    )


@patch(
    "gravitorch.datapipes.iter.batching.torch.randperm",
    lambda *args, **kwargs: torch.tensor([5, 4, 6, 3, 7, 2, 8, 1, 9, 0]),
)
def test_tuple_batcher_iter_shuffle_true() -> None:
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
            )
        ),
        (
            (
                torch.tensor([[10, 11], [8, 9], [12, 13], [6, 7]]),
                torch.tensor([[5], [4], [6], [3]]),
            ),
            (
                torch.tensor([[14, 15], [4, 5], [16, 17], [2, 3]]),
                torch.tensor([[7], [2], [8], [1]]),
            ),
            (torch.tensor([[18, 19], [0, 1]]), torch.tensor([[9], [0]])),
        ),
    )


def test_tuple_batcher_generator_same_random_seed() -> None:
    assert objects_are_equal(
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
    )


def test_tuple_batcher_generator_different_random_seeds() -> None:
    assert not objects_are_equal(
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=1,
            )
        ),
        tuple(
            TupleBatcher(
                [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                batch_size=4,
                shuffle=True,
                random_seed=2,
            )
        ),
    )


def test_tuple_batcher_generator_repeat() -> None:
    datapipe = TupleBatcher(
        [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=4,
        shuffle=True,
    )
    assert not objects_are_equal(tuple(datapipe), tuple(datapipe))


def test_tuple_batcher_len_datapipe() -> None:
    with raises(TypeError, match="TupleBatcherIterDataPipe instance doesn't have valid length"):
        len(TupleBatcher(SourceWrapper([]), batch_size=4))


def test_tuple_batcher_len_batch_size_2() -> None:
    assert len(TupleBatcher([torch.ones(10), torch.zeros(10)], batch_size=2)) == 5


def test_tuple_batcher_len_batch_size_4() -> None:
    assert len(TupleBatcher([torch.ones(10), torch.zeros(10)], batch_size=4)) == 3


def test_tuple_batcher_getstate() -> None:
    state = TupleBatcher(
        [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
        batch_size=2,
    ).__getstate__()
    assert len(state) == 4
    assert objects_are_equal(
        state["_datapipe_or_data"], [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)]
    )
    assert state["_batch_size"] == 2
    assert not state["_shuffle"]
    assert isinstance(state["_generator"], Tensor)


def test_tuple_batcher_setstate() -> None:
    state = torch.Generator().get_state()
    dp = TupleBatcher([], batch_size=2)
    dp.__setstate__(
        {
            "_datapipe_or_data": [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
            "_batch_size": 4,
            "_shuffle": True,
            "_generator": state,
        }
    )
    assert objects_are_equal(
        dp._datapipe_or_data, [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)]
    )
    assert dp._batch_size == 4
    assert dp._shuffle
    assert dp._generator.get_state().equal(state)


def test_tuple_batcher_state_repeat() -> None:
    dp = TupleBatcher(
        [torch.arange(40).view(20, 2), torch.arange(20).view(20, 1)],
        batch_size=4,
        shuffle=True,
    )
    state = dp.__getstate__()
    batches1 = tuple(dp)
    batches2 = tuple(dp)
    dp.__setstate__(state)
    batches3 = tuple(dp)
    assert not objects_are_equal(batches1, batches2)
    assert objects_are_equal(batches1, batches3)


@torchdata_available
def test_tuple_batcher_dataloader2_data() -> None:
    assert objects_are_equal(
        tuple(
            DataLoader2(
                TupleBatcher(
                    [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                    batch_size=2,
                )
            )
        ),
        (
            (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0], [1]])),
            (torch.tensor([[4, 5], [6, 7]]), torch.tensor([[2], [3]])),
            (torch.tensor([[8, 9], [10, 11]]), torch.tensor([[4], [5]])),
            (torch.tensor([[12, 13], [14, 15]]), torch.tensor([[6], [7]])),
            (torch.tensor([[16, 17], [18, 19]]), torch.tensor([[8], [9]])),
        ),
    )


@torchdata_available
def test_tuple_batcher_dataloader2_datapipe() -> None:
    assert objects_are_equal(
        tuple(
            DataLoader2(
                TupleBatcher(
                    IterableWrapper(
                        [
                            [torch.arange(20).view(10, 2), torch.arange(10).view(10, 1)],
                            [torch.arange(10).view(5, 2), torch.arange(5).view(5, 1)],
                        ]
                    ),
                    batch_size=2,
                )
            )
        ),
        (
            (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0], [1]])),
            (torch.tensor([[4, 5], [6, 7]]), torch.tensor([[2], [3]])),
            (torch.tensor([[8, 9], [10, 11]]), torch.tensor([[4], [5]])),
            (torch.tensor([[12, 13], [14, 15]]), torch.tensor([[6], [7]])),
            (torch.tensor([[16, 17], [18, 19]]), torch.tensor([[8], [9]])),
            (torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0], [1]])),
            (torch.tensor([[4, 5], [6, 7]]), torch.tensor([[2], [3]])),
            (torch.tensor([[8, 9]]), torch.tensor([[4]])),
        ),
    )
