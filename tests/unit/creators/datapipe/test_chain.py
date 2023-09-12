from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher, Multiplexer

from gravitorch.creators.datapipe import ChainedDataPipeCreator
from gravitorch.datapipes.iter import SourceWrapper

############################################
#     Tests for ChainedDataPipeCreator     #
############################################


def test_chained_datapipe_creator_str() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert str(creator).startswith("ChainedDataPipeCreator(")


def test_chained_datapipe_creator_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because config is empty"
    ):
        ChainedDataPipeCreator([])


def test_chained_datapipe_creator_dict() -> None:
    creator = ChainedDataPipeCreator(
        {
            OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_dict_source_inputs() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_dict_one_input_datapipe() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_dict_two_input_datapipes() -> None:
    creator = ChainedDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
    )
    datapipe = creator.create(
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_chained_datapipe_creator_sequence_1() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_chained_datapipe_creator_sequence_2() -> None:
    creator = ChainedDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_source_inputs() -> None:
    creator = ChainedDataPipeCreator(
        config=[
            {OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_source_inputs_datapipe() -> None:
    creator = ChainedDataPipeCreator(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_chained_datapipe_creator_sequence_multiple_input_datapipes() -> None:
    creator = ChainedDataPipeCreator(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ]
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])
