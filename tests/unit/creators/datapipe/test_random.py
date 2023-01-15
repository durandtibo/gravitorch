from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import raises
from torch.utils.data import IterDataPipe

from gravitorch.creators.datapipe import EpochRandomIterDataPipeCreator
from gravitorch.engines import BaseEngine

####################################################
#     Tests for EpochRandomIterDataPipeCreator     #
####################################################


def test_epoch_random_iter_data_pipe_creator_str():
    assert str(EpochRandomIterDataPipeCreator({})).startswith("EpochRandomIterDataPipeCreator(")


def test_epoch_random_iter_data_pipe_creator_create_engine_none():
    with raises(ValueError):
        EpochRandomIterDataPipeCreator({}).create()


@patch("gravitorch.creators.datapipe.random.dist.get_rank", lambda *args, **kwargs: 0)
def test_epoch_random_iter_data_pipe_creator_create():
    datapipe = Mock(spec=IterDataPipe)
    factory_mock = Mock(return_value=datapipe)
    with patch("gravitorch.creators.datapipe.random.factory", factory_mock):
        creator = EpochRandomIterDataPipeCreator({OBJECT_TARGET: "MyIterDataPipe", "key": "value"})
        assert (
            creator.create(engine=Mock(spec=BaseEngine, epoch=1, max_epochs=10, random_seed=42))
            == datapipe
        )
        factory_mock.assert_called_once_with("MyIterDataPipe", key="value", random_seed=43)


@patch("gravitorch.creators.datapipe.random.dist.get_rank", lambda *args, **kwargs: 0)
def test_epoch_random_iter_data_pipe_creator_create_with_source_inputs():
    datapipe = Mock(spec=IterDataPipe)
    factory_mock = Mock(return_value=datapipe)
    with patch("gravitorch.creators.datapipe.random.factory", factory_mock):
        creator = EpochRandomIterDataPipeCreator({OBJECT_TARGET: "MyIterDataPipe", "key": "value"})
        assert (
            creator.create(
                engine=Mock(spec=BaseEngine, epoch=1, max_epochs=10, random_seed=42),
                source_inputs=("my_input",),
            )
            == datapipe
        )
        factory_mock.assert_called_once_with(
            "MyIterDataPipe", "my_input", key="value", random_seed=43
        )


@patch("gravitorch.creators.datapipe.random.dist.get_rank", lambda *args, **kwargs: 0)
def test_epoch_random_iter_data_pipe_creator_create_rank_0():
    assert (
        EpochRandomIterDataPipeCreator(
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.TensorDictShufflerIterDataPipe",
                "source_datapipe": Mock(spec=IterDataPipe),
                "random_seed": 42,
            }
        )
        .create(engine=Mock(spec=BaseEngine, epoch=1, max_epochs=10, random_seed=42))
        .random_seed
        == 43
    )


@patch("gravitorch.creators.datapipe.random.dist.get_rank", lambda *args, **kwargs: 1)
def test_epoch_random_iter_data_pipe_creator_get_random_seed_rank_1():
    assert (
        EpochRandomIterDataPipeCreator(
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.TensorDictShufflerIterDataPipe",
                "source_datapipe": Mock(spec=IterDataPipe),
                "random_seed": 42,
            }
        )
        .create(engine=Mock(spec=BaseEngine, epoch=1, max_epochs=10, random_seed=42))
        .random_seed
        == 53
    )


@patch("gravitorch.creators.datapipe.random.dist.get_rank", lambda *args, **kwargs: 0)
def test_epoch_random_iter_data_pipe_creator_create_no_random_seed():
    assert (
        EpochRandomIterDataPipeCreator(
            {
                OBJECT_TARGET: "gravitorch.data.datapipes.iter.TensorDictShufflerIterDataPipe",
                "source_datapipe": Mock(spec=IterDataPipe),
            }
        )
        .create(engine=Mock(spec=BaseEngine, epoch=2, max_epochs=10, random_seed=35))
        .random_seed
        == 37
    )
