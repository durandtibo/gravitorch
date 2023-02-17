import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises

from gravitorch.creators.datapipe import (
    BaseIterDataPipeCreator,
    SequentialIterDataPipeCreator,
)
from gravitorch.data.datacreators import BaseDataCreator, HypercubeVertexDataCreator
from gravitorch.data.datapipes.iter import SourceWrapper
from gravitorch.datasources import (
    DataCreatorIterDataPipeCreatorDataSource,
    IterDataPipeCreatorDataSource,
    LoaderNotFoundError,
)
from gravitorch.engines import BaseEngine
from gravitorch.utils.asset import AssetNotFoundError

###################################################
#     Tests for IterDataPipeCreatorDataSource     #
###################################################


@fixture
def data_source() -> IterDataPipeCreatorDataSource:
    return IterDataPipeCreatorDataSource(
        {
            "train": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                        "source": [1, 2, 3, 4],
                    },
                ],
            },
            "val": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialIterDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper",
                        "source": ["a", "b", "c"],
                    },
                ],
            },
        }
    )


def test_iter_data_pipe_creator_data_source_str():
    assert str(
        IterDataPipeCreatorDataSource({"train": Mock(spec=BaseIterDataPipeCreator)})
    ).startswith("IterDataPipeCreatorDataSource(")


def test_iter_data_pipe_creator_data_source_attach(
    caplog: LogCaptureFixture, data_source: IterDataPipeCreatorDataSource
):
    with caplog.at_level(logging.INFO):
        data_source.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_iter_data_pipe_creator_data_source_get_asset_exists(
    data_source: IterDataPipeCreatorDataSource,
):
    data_source._asset_manager.add_asset("something", 2)
    assert data_source.get_asset("something") == 2


def test_iter_data_pipe_creator_data_source_get_asset_does_not_exist(
    data_source: IterDataPipeCreatorDataSource,
):
    with raises(AssetNotFoundError):
        data_source.get_asset("something")


def test_iter_data_pipe_creator_data_source_has_asset_true(
    data_source: IterDataPipeCreatorDataSource,
):
    data_source._asset_manager.add_asset("something", 1)
    assert data_source.has_asset("something")


def test_iter_data_pipe_creator_data_source_has_asset_false(
    data_source: IterDataPipeCreatorDataSource,
):
    assert not data_source.has_asset("something")


def test_iter_data_pipe_creator_data_source_get_data_loader_train(
    data_source: IterDataPipeCreatorDataSource,
):
    loader = data_source.get_data_loader("train")
    assert isinstance(loader, SourceWrapper)
    assert tuple(loader) == (1, 2, 3, 4)


def test_iter_data_pipe_creator_data_source_get_data_loader_val(
    data_source: IterDataPipeCreatorDataSource,
):
    loader = data_source.get_data_loader("val")
    assert isinstance(loader, SourceWrapper)
    assert tuple(loader) == ("a", "b", "c")


def test_iter_data_pipe_creator_data_source_get_data_loader_missing(
    data_source: IterDataPipeCreatorDataSource,
):
    with raises(LoaderNotFoundError):
        data_source.get_data_loader("missing")


def test_iter_data_pipe_creator_data_source_get_data_loader_with_engine():
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    data_source = IterDataPipeCreatorDataSource({"train": datapipe_creator})
    data_source.get_data_loader("train", engine=engine)
    datapipe_creator.create.assert_called_once_with(engine=engine)


def test_iter_data_pipe_creator_data_source_get_data_loader_without_engine():
    datapipe_creator = Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    data_source = IterDataPipeCreatorDataSource({"train": datapipe_creator})
    data_source.get_data_loader("train")
    datapipe_creator.create.assert_called_once_with(engine=None)


def test_iter_data_pipe_creator_data_source_has_data_loader_true(
    data_source: IterDataPipeCreatorDataSource,
):
    assert data_source.has_data_loader("train")


def test_iter_data_pipe_creator_data_source_has_data_loader_false(
    data_source: IterDataPipeCreatorDataSource,
):
    assert not data_source.has_data_loader("missing")


def test_iter_data_pipe_creator_data_source_load_state_dict(
    data_source: IterDataPipeCreatorDataSource,
):
    data_source.load_state_dict({})


def test_iter_data_pipe_creator_data_source_state_dict(data_source: IterDataPipeCreatorDataSource):
    assert data_source.state_dict() == {}


##############################################################
#     Tests for DataCreatorIterDataPipeCreatorDataSource     #
##############################################################


def test_data_creator_iter_data_pipe_creator_data_source_str():
    assert str(
        DataCreatorIterDataPipeCreatorDataSource(
            datapipe_creators={"train": Mock(spec=BaseIterDataPipeCreator)},
            data_creators={"train": Mock(spec=BaseDataCreator)},
        )
    ).startswith("DataCreatorIterDataPipeCreatorDataSource(")


def test_data_creator_iter_data_pipe_creator_data_source_data_creators():
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": Mock(spec=BaseIterDataPipeCreator)},
        data_creators={
            "train": HypercubeVertexDataCreator(num_examples=10, num_classes=5),
            "val": {
                OBJECT_TARGET: "gravitorch.data.datacreators.HypercubeVertexDataCreator",
                "num_examples": 10,
                "num_classes": 5,
            },
        },
    )
    assert len(creator._data_creators) == 2
    isinstance(creator._data_creators["train"], HypercubeVertexDataCreator)
    isinstance(creator._data_creators["val"], HypercubeVertexDataCreator)


def test_data_creator_iter_data_pipe_creator_data_source_create_datapipe():
    data_creator = Mock(spec=BaseDataCreator, create=Mock(return_value=[1, 2, 3]))
    datapipe_creator = Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={"train": data_creator},
    )
    assert creator._create_datapipe("train") == ["a", "b", "c"]
    data_creator.create.assert_called_once_with()
    assert creator._data == {"train": [1, 2, 3]}
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=([1, 2, 3],))


def test_data_creator_iter_data_pipe_creator_data_source_create_datapipe_no_data_creator():
    datapipe_creator = Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train")
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_iter_data_pipe_creator_data_source_create_datapipe_no_data_creator_with_engine():
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseIterDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train", engine)
    datapipe_creator.create.assert_called_once_with(engine=engine, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_iter_data_pipe_creator_data_source_get_data_loader():
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={
            "train": SequentialIterDataPipeCreator(
                config=[
                    {OBJECT_TARGET: "gravitorch.data.datapipes.iter.SourceWrapper"},
                ]
            )
        },
        data_creators={
            "train": HypercubeVertexDataCreator(num_examples=10, num_classes=5),
        },
    )
    datapipe = creator.get_data_loader("train")
    assert isinstance(datapipe, SourceWrapper)
