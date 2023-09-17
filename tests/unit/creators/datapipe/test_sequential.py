from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher

from gravitorch.creators.datapipe import (
    BaseDataPipeCreator,
    ChainedDataPipeCreator,
    SequentialDataPipeCreator,
)
from gravitorch.engines import BaseEngine

###############################################
#     Tests for SequentialDataPipeCreator     #
###############################################


def test_sequential_datapipe_creator_str() -> None:
    assert str(SequentialDataPipeCreator([Mock(spec=BaseDataPipeCreator)])).startswith(
        "SequentialDataPipeCreator("
    )


def test_sequential_datapipe_creator_creators() -> None:
    creator = SequentialDataPipeCreator(
        [
            ChainedDataPipeCreator(
                {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                    "source": [1, 2, 3, 4],
                },
            ),
            {
                OBJECT_TARGET: "gravitorch.creators.datapipe.ChainedDataPipeCreator",
                "config": {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
                    "batch_size": 2,
                },
            },
        ]
    )
    assert len(creator._creators) == 2
    assert isinstance(creator._creators[0], ChainedDataPipeCreator)
    assert isinstance(creator._creators[1], ChainedDataPipeCreator)


def test_sequential_datapipe_creator_creators_empty() -> None:
    with raises(
        ValueError, match="It is not possible to create a DataPipe because creators is empty"
    ):
        SequentialDataPipeCreator([])


def test_sequential_datapipe_creator_create_1() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output")),
    ]
    creator = SequentialDataPipeCreator(creators)
    assert creator.create() == "output"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)


def test_sequential_datapipe_creator_create_2() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialDataPipeCreator(creators)
    assert creator.create() == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_datapipe_creator_create_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialDataPipeCreator(creators)
    assert creator.create(engine) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_datapipe_creator_create_with_source_inputs() -> None:
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialDataPipeCreator(creators)
    assert creator.create(source_inputs=("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_datapipe_creator_create_with_engine_and_source_inputs() -> None:
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialDataPipeCreator(creators)
    assert creator.create(engine, ("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_datapipe_creator_create_batcher() -> None:
    creator = SequentialDataPipeCreator(
        [
            ChainedDataPipeCreator(
                {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                    "iterable": [1, 2, 3, 4],
                },
            ),
            ChainedDataPipeCreator(
                {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
            ),
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])
