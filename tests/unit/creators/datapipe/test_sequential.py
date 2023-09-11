from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher, Multiplexer

from gravitorch.creators.datapipe import (
    BaseDataPipeCreator,
    SequentialCreatorIterDataPipeCreator,
    SequentialDataPipeCreator,
)
from gravitorch.datapipes.iter import SourceWrapper
from gravitorch.engines import BaseEngine

###############################################
#     Tests for SequentialDataPipeCreator     #
###############################################


def test_sequential_datapipe_creator_str() -> None:
    creator = SequentialDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert str(creator).startswith("SequentialDataPipeCreator(")


def test_sequential_datapipe_creator_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because config is empty"
    ):
        SequentialDataPipeCreator([])


def test_sequential_datapipe_creator_dict() -> None:
    creator = SequentialDataPipeCreator(
        {
            OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_datapipe_creator_dict_source_inputs() -> None:
    creator = SequentialDataPipeCreator(
        config={OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_datapipe_creator_dict_one_input_datapipe() -> None:
    creator = SequentialDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_dict_two_input_datapipes() -> None:
    creator = SequentialDataPipeCreator(
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


def test_sequential_datapipe_creator_sequence_1() -> None:
    creator = SequentialDataPipeCreator(
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


def test_sequential_datapipe_creator_sequence_2() -> None:
    creator = SequentialDataPipeCreator(
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


def test_sequential_datapipe_creator_sequence_source_inputs() -> None:
    creator = SequentialDataPipeCreator(
        config=[
            {OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_sequence_source_inputs_datapipe() -> None:
    creator = SequentialDataPipeCreator(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_datapipe_creator_sequence_multiple_input_datapipes() -> None:
    creator = SequentialDataPipeCreator(
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


##########################################################
#     Tests for SequentialCreatorIterDataPipeCreator     #
##########################################################


def test_sequential_creator_iter_datapipe_creator_str() -> None:
    assert str(SequentialCreatorIterDataPipeCreator([Mock(spec=BaseDataPipeCreator)])).startswith(
        "SequentialCreatorIterDataPipeCreator("
    )


def test_sequential_creator_iter_datapipe_creator_creators() -> None:
    creator = SequentialCreatorIterDataPipeCreator(
        [
            SequentialDataPipeCreator(
                {
                    OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                    "source": [1, 2, 3, 4],
                },
            ),
            {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialDataPipeCreator",
                "config": {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
                    "batch_size": 2,
                },
            },
        ]
    )
    assert len(creator._creators) == 2
    assert isinstance(creator._creators[0], SequentialDataPipeCreator)
    assert isinstance(creator._creators[1], SequentialDataPipeCreator)


def test_sequential_creator_iter_datapipe_creator_creators_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because creators is empty"
    ):
        SequentialCreatorIterDataPipeCreator([])


def test_sequential_creator_iter_datapipe_creator_create_1() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create() == "output"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)


def test_sequential_creator_iter_datapipe_creator_create_2() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create() == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(engine) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_source_inputs() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(source_inputs=("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_engine_and_source_inputs() -> None:
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(engine, ("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_batcher() -> None:
    creator = SequentialCreatorIterDataPipeCreator(
        [
            SequentialDataPipeCreator(
                {
                    OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                    "source": [1, 2, 3, 4],
                },
            ),
            SequentialDataPipeCreator(
                {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
            ),
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])
