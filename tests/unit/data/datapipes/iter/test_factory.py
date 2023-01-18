from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher

from gravitorch.data.datapipes.iter import SourceWrapper, setup_iter_datapipe
from gravitorch.data.datapipes.iter.factory import create_sequential_iter_datapipe

#####################################################
#     Tests for create_sequential_iter_datapipe     #
#####################################################


def test_create_sequential_iter_datapipe_empty():
    with raises(ValueError):
        create_sequential_iter_datapipe([])


def test_create_sequential_iter_datapipe_1():
    datapipe = create_sequential_iter_datapipe(
        [{OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper", "source": [1, 2, 3, 4]}]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iter_datapipe_2():
    datapipe = create_sequential_iter_datapipe(
        [
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


#########################################
#     Tests for setup_iter_datapipe     #
#########################################


def test_setup_iter_datapipe_object():
    datapipe = SourceWrapper([1, 2, 3, 4])
    assert setup_iter_datapipe(datapipe) is datapipe


def test_setup_iter_datapipe_sequence():
    datapipe = setup_iter_datapipe(
        [{OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper", "source": [1, 2, 3, 4]}]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)
