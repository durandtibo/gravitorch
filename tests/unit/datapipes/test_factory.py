from __future__ import annotations

from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.utils.data.datapipes.iter import Batcher, IterableWrapper, Multiplexer
from torch.utils.data.datapipes.map import SequenceWrapper

from gravitorch.datapipes import (
    create_chained_datapipe,
    is_datapipe_config,
    setup_datapipe,
)

#############################################
#     Tests for create_chained_datapipe     #
#############################################


@mark.parametrize("config", (list(), tuple(), dict()))
def test_create_chained_datapipe_empty(config: list | tuple | dict) -> None:
    with raises(
        RuntimeError, match="It is not possible to create a DataPipe because the config is empty"
    ):
        create_chained_datapipe(config)


def test_create_chained_datapipe_dict() -> None:
    datapipe = create_chained_datapipe(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4],
        }
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_dict_source_inputs() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, IterableWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_chained_datapipe_dict_one_input_datapipe() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        source_inputs=[IterableWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_dict_two_input_datapipes() -> None:
    datapipe = create_chained_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_create_chained_datapipe_sequence_1() -> None:
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


def test_create_chained_datapipe_sequence_2() -> None:
    datapipe = create_chained_datapipe(
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


def test_create_chained_datapipe_sequence_source_inputs() -> None:
    datapipe = create_chained_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_sequence_source_inputs_datapipe() -> None:
    datapipe = create_chained_datapipe(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
        source_inputs=[IterableWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_chained_datapipe_sequence_multiple_input_datapipes() -> None:
    datapipe = create_chained_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=[
            IterableWrapper([1, 2, 3, 4]),
            IterableWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])


########################################
#     Tests for is_datapipe_config     #
########################################


def test_is_datapipe_config_true_iter() -> None:
    assert is_datapipe_config(
        {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper", "iterable": [1, 2, 3, 4]}
    )


def test_is_datapipe_config_true_map() -> None:
    assert is_datapipe_config(
        {OBJECT_TARGET: "torch.utils.data.datapipes.map.SequenceWrapper", "sequence": [1, 2, 3, 4]}
    )


def test_is_datapipe_config_false() -> None:
    assert not is_datapipe_config({OBJECT_TARGET: "torch.nn.Identity"})


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
