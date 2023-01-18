from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data.datapipes.iter import Batcher, Multiplexer

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    SequentialCreatorIterDataPipeCreator,
    SequentialIterDataPipeCreator,
    create_sequential_iter_datapipe,
)
from gravitorch.data.datapipes.iter import SourceWrapper
from gravitorch.engines import BaseEngine

###################################################
#     Tests for SequentialIterDataPipeCreator     #
###################################################


def test_sequential_iter_datapipe_creator_str():
    creator = SequentialIterDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert str(creator).startswith("SequentialIterDataPipeCreator(")


def test_sequential_iter_datapipe_creator_empty():
    with raises(ValueError):
        SequentialIterDataPipeCreator([])


def test_sequential_iter_datapipe_creator_dict():
    creator = SequentialIterDataPipeCreator(
        {
            OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_iter_datapipe_creator_dict_source_inputs():
    creator = SequentialIterDataPipeCreator(
        config={OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_iter_datapipe_creator_dict_one_input_datapipe():
    creator = SequentialIterDataPipeCreator(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_iter_datapipe_creator_dict_two_input_datapipes():
    creator = SequentialIterDataPipeCreator(
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


def test_sequential_iter_datapipe_creator_sequence_1():
    creator = SequentialIterDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_sequential_iter_datapipe_creator_sequence_2():
    creator = SequentialIterDataPipeCreator(
        [
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            },
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_iter_datapipe_creator_sequence_source_inputs():
    creator = SequentialIterDataPipeCreator(
        config=[
            {OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
    )
    datapipe = creator.create(source_inputs=([1, 2, 3, 4],))
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_iter_datapipe_creator_sequence_source_inputs_datapipe():
    creator = SequentialIterDataPipeCreator(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
    )
    datapipe = creator.create(source_inputs=[SourceWrapper([1, 2, 3, 4])])
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_sequential_iter_datapipe_creator_sequence_multiple_input_datapipes():
    creator = SequentialIterDataPipeCreator(
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


#####################################################
#     Tests for create_sequential_iter_datapipe     #
#####################################################


def test_create_sequential_iter_datapipe_empty_dict():
    with raises(ValueError):
        create_sequential_iter_datapipe({})


def test_create_sequential_iter_datapipe_empty_sequence():
    with raises(ValueError):
        create_sequential_iter_datapipe([])


def test_create_sequential_iter_datapipe_dict():
    datapipe = create_sequential_iter_datapipe(
        {
            OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
            "source": [1, 2, 3, 4],
        }
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iter_datapipe_dict_source_inputs():
    datapipe = create_sequential_iter_datapipe(
        config={OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iter_datapipe_dict_one_input_datapipe():
    datapipe = create_sequential_iter_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        source_inputs=[SourceWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_iter_datapipe_dict_two_input_datapipes():
    datapipe = create_sequential_iter_datapipe(
        config={OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Multiplexer)
    assert tuple(datapipe) == (1, 11, 2, 12, 3, 13, 4, 14)


def test_create_sequential_iter_datapipe_sequence_1():
    datapipe = create_sequential_iter_datapipe(
        [
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                "source": [1, 2, 3, 4],
            }
        ]
    )
    assert isinstance(datapipe, SourceWrapper)
    assert tuple(datapipe) == (1, 2, 3, 4)


def test_create_sequential_iter_datapipe_sequence_2():
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


def test_create_sequential_iter_datapipe_sequence_source_inputs():
    datapipe = create_sequential_iter_datapipe(
        config=[
            {OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=([1, 2, 3, 4],),
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_iter_datapipe_sequence_source_inputs_datapipe():
    datapipe = create_sequential_iter_datapipe(
        config=[{OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2}],
        source_inputs=[SourceWrapper([1, 2, 3, 4])],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])


def test_create_sequential_iter_datapipe_sequence_multiple_input_datapipes():
    datapipe = create_sequential_iter_datapipe(
        config=[
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Multiplexer"},
            {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
        ],
        source_inputs=[
            SourceWrapper([1, 2, 3, 4]),
            SourceWrapper([11, 12, 13, 14]),
        ],
    )
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 11], [2, 12], [3, 13], [4, 14])


##########################################################
#     Tests for SequentialCreatorIterDataPipeCreator     #
##########################################################


def test_sequential_creator_iter_datapipe_creator_str():
    assert str(
        SequentialCreatorIterDataPipeCreator([Mock(spec=BaseIterDataPipeCreator)])
    ).startswith("SequentialCreatorIterDataPipeCreator(")


def test_sequential_creator_iter_datapipe_creator_creators():
    creator = SequentialCreatorIterDataPipeCreator(
        [
            SequentialIterDataPipeCreator(
                {
                    OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                    "source": [1, 2, 3, 4],
                },
            ),
            {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                "config": {
                    OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher",
                    "batch_size": 2,
                },
            },
        ]
    )
    assert len(creator._creators) == 2
    assert isinstance(creator._creators[0], SequentialIterDataPipeCreator)
    assert isinstance(creator._creators[1], SequentialIterDataPipeCreator)


def test_sequential_creator_iter_datapipe_creator_creators_empty():
    with raises(ValueError):
        SequentialCreatorIterDataPipeCreator([])


def test_sequential_creator_iter_datapipe_creator_create_1():
    creators = [
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create() == "output"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)


def test_sequential_creator_iter_datapipe_creator_create_2():
    creators = [
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create() == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_engine():
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(engine) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=None)
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_source_inputs():
    creators = [
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(source_inputs=("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=None, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=None, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_with_engine_and_source_inputs():
    engine = Mock(spec=BaseEngine)
    creators = [
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output1")),
        Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value="output2")),
    ]
    creator = SequentialCreatorIterDataPipeCreator(creators)
    assert creator.create(engine, ("my_input",)) == "output2"
    creators[0].create.assert_called_once_with(engine=engine, source_inputs=("my_input",))
    creators[1].create.assert_called_once_with(engine=engine, source_inputs=("output1",))


def test_sequential_creator_iter_datapipe_creator_create_batcher():
    creator = SequentialCreatorIterDataPipeCreator(
        [
            SequentialIterDataPipeCreator(
                {
                    OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                    "source": [1, 2, 3, 4],
                },
            ),
            SequentialIterDataPipeCreator(
                {OBJECT_TARGET: "torch.utils.data.datapipes.iter.Batcher", "batch_size": 2},
            ),
        ]
    )
    datapipe = creator.create()
    assert isinstance(datapipe, Batcher)
    assert tuple(datapipe) == ([1, 2], [3, 4])
