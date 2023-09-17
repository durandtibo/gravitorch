from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher, IterableWrapper

from gravitorch.datapipes.iter import (
    create_sequential_iter_datapipe,
    is_iter_datapipe_config,
    setup_iter_datapipe,
)

#####################################################
#     Tests for create_sequential_iter_datapipe     #
#####################################################


def test_create_sequential_iter_datapipe_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because the configs are empty"
    ):
        create_sequential_iter_datapipe([])


def test_create_sequential_iter_datapipe_1() -> None:
    datapipe = create_sequential_iter_datapipe(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iter_datapipe_2() -> None:
    datapipe = create_sequential_iter_datapipe(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


#############################################
#     Tests for is_iter_datapipe_config     #
#############################################


def test_is_iter_datapipe_config_true() -> None:
    assert is_iter_datapipe_config(
        {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": [1, 2, 3, 4]}
    )


def test_is_iter_datapipe_config_false() -> None:
    assert not is_iter_datapipe_config({OBJECT_TARGET: "torch.nn.Identity"})


#########################################
#     Tests for setup_iter_datapipe     #
#########################################


def test_setup_iter_datapipe_object() -> None:
    datapipe = IterableWrapper([1, 2, 3, 4])
    assert setup_iter_datapipe(datapipe) is datapipe


def test_setup_iter_datapipe_sequence() -> None:
    datapipe = setup_iter_datapipe(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4],
        }
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)
