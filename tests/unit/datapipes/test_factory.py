from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher, IterableWrapper
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.datapipes import (
    create_chained_datapipe,
    is_datapipe_config,
    setup_datapipe,
)

#############################################
#     Tests for create_chained_datapipe     #
#############################################


def test_create_chained_datapipe_empty() -> None:
    with raises(
        RuntimeError, match="It is not possible to create a DataPipe because the configs are empty"
    ):
        create_chained_datapipe([])


def test_create_chained_datapipe_one() -> None:
    datapipe = create_chained_datapipe(
        [
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4],
            }
        ]
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_two() -> None:
    datapipe = create_chained_datapipe(
        [
            {
                "_target_": "torch.utils.data.datapipes.map.SequenceWrapper",
                "sequence": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


########################################
#     Tests for is_datapipe_config     #
########################################


def test_is_datapipe_config_true_iter() -> None:
    assert is_datapipe_config(
        {"_target_": "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": [1, 2, 3, 4]}
    )


def test_is_datapipe_config_true_map() -> None:
    assert is_datapipe_config(
        {"_target_": "torch.utils.data.datapipes.map.SequenceWrapper", "sequence": [1, 2, 3, 4]}
    )


def test_is_datapipe_config_false() -> None:
    assert not is_datapipe_config({"_target_": "torch.nn.Identity"})


####################################
#     Tests for setup_datapipe     #
####################################


def test_setup_datapipe_object_iter() -> None:
    datapipe = IterableWrapper((1, 2, 3, 4, 5))
    assert setup_datapipe(datapipe) is datapipe


def test_setup_datapipe_object_map() -> None:
    datapipe = SequenceWrapper((1, 2, 3, 4, 5))
    assert setup_datapipe(datapipe) is datapipe


def test_setup_datapipe_config_iter() -> None:
    assert isinstance(
        setup_datapipe(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": (1, 2, 3, 4, 5),
            }
        ),
        IterableWrapper,
    )


def test_setup_datapipe_config_map() -> None:
    assert isinstance(
        setup_datapipe(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.map.SequenceWrapper",
                "sequence": (1, 2, 3, 4, 5),
            }
        ),
        SequenceWrapper,
    )
