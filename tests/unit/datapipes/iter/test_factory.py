from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher

from gravitorch.datapipes.iter import SourceWrapper, setup_iterdatapipe
from gravitorch.datapipes.iter.factory import (
    create_sequential_iterdatapipe,
    is_iterdatapipe_config,
)

####################################################
#     Tests for create_sequential_iterdatapipe     #
####################################################


def test_create_sequential_iterdatapipe_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because the configs are empty"
    ):
        create_sequential_iterdatapipe([])


def test_create_sequential_iterdatapipe_1() -> None:
    datapipe = create_sequential_iterdatapipe(
        [{OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper", "source": [1, 2, 3, 4]}]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iterdatapipe_2() -> None:
    datapipe = create_sequential_iterdatapipe(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


############################################
#     Tests for is_iterdatapipe_config     #
############################################


def test_is_iterdatapipe_config_true() -> None:
    assert is_iterdatapipe_config(
        {"_target_": "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": [1, 2, 3, 4]}
    )


def test_is_iterdatapipe_config_false() -> None:
    assert not is_iterdatapipe_config({"_target_": "torch.nn.Identity"})


########################################
#     Tests for setup_iterdatapipe     #
########################################


def test_setup_iterdatapipe_object() -> None:
    datapipe = SourceWrapper([1, 2, 3, 4])
    assert setup_iterdatapipe(datapipe) is datapipe


def test_setup_iterdatapipe_sequence() -> None:
    datapipe = setup_iterdatapipe(
        [{OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper", "source": [1, 2, 3, 4]}]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)
