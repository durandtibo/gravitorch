import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises

from gravitorch.creators.datapipe import BaseDataPipeCreator, SequentialDataPipeCreator
from gravitorch.data.datacreators import BaseDataCreator, HypercubeVertexDataCreator
from gravitorch.datapipes.iter import SourceWrapper
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
def datasource() -> IterDataPipeCreatorDataSource:
    return IterDataPipeCreatorDataSource(
        {
            "train": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                        "source": [1, 2, 3, 4],
                    },
                ],
            },
            "val": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.SequentialDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper",
                        "source": ["a", "b", "c"],
                    },
                ],
            },
        }
    )


def test_iter_data_pipe_creator_datasource_str() -> None:
    assert str(IterDataPipeCreatorDataSource({"train": Mock(spec=BaseDataPipeCreator)})).startswith(
        "IterDataPipeCreatorDataSource("
    )


def test_iter_data_pipe_creator_datasource_attach(
    caplog: LogCaptureFixture, datasource: IterDataPipeCreatorDataSource
) -> None:
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_iter_data_pipe_creator_datasource_get_asset_exists(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_iter_data_pipe_creator_datasource_get_asset_does_not_exist(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_iter_data_pipe_creator_datasource_has_asset_true(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_iter_data_pipe_creator_datasource_has_asset_false(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    assert not datasource.has_asset("something")


def test_iter_data_pipe_creator_datasource_get_dataloader_train(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    loader = datasource.get_dataloader("train")
    assert isinstance(loader, SourceWrapper)
    assert tuple(loader) == (1, 2, 3, 4)


def test_iter_data_pipe_creator_datasource_get_dataloader_val(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    loader = datasource.get_dataloader("val")
    assert isinstance(loader, SourceWrapper)
    assert tuple(loader) == ("a", "b", "c")


def test_iter_data_pipe_creator_datasource_get_dataloader_missing(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    with raises(LoaderNotFoundError):
        datasource.get_dataloader("missing")


def test_iter_data_pipe_creator_datasource_get_dataloader_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = IterDataPipeCreatorDataSource({"train": datapipe_creator})
    datasource.get_dataloader("train", engine=engine)
    datapipe_creator.create.assert_called_once_with(engine=engine)


def test_iter_data_pipe_creator_datasource_get_dataloader_without_engine() -> None:
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = IterDataPipeCreatorDataSource({"train": datapipe_creator})
    datasource.get_dataloader("train")
    datapipe_creator.create.assert_called_once_with(engine=None)


def test_iter_data_pipe_creator_datasource_has_dataloader_true(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    assert datasource.has_dataloader("train")


def test_iter_data_pipe_creator_datasource_has_dataloader_false(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    assert not datasource.has_dataloader("missing")


def test_iter_data_pipe_creator_datasource_load_state_dict(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    datasource.load_state_dict({})


def test_iter_data_pipe_creator_datasource_state_dict(
    datasource: IterDataPipeCreatorDataSource,
) -> None:
    assert datasource.state_dict() == {}


##############################################################
#     Tests for DataCreatorIterDataPipeCreatorDataSource     #
##############################################################


def test_data_creator_iter_data_pipe_creator_datasource_str() -> None:
    assert str(
        DataCreatorIterDataPipeCreatorDataSource(
            datapipe_creators={"train": Mock(spec=BaseDataPipeCreator)},
            data_creators={"train": Mock(spec=BaseDataCreator)},
        )
    ).startswith("DataCreatorIterDataPipeCreatorDataSource(")


def test_data_creator_iter_data_pipe_creator_datasource_data_creators() -> None:
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": Mock(spec=BaseDataPipeCreator)},
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


def test_data_creator_iter_data_pipe_creator_datasource_create_datapipe() -> None:
    data_creator = Mock(spec=BaseDataCreator, create=Mock(return_value=[1, 2, 3]))
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={"train": data_creator},
    )
    assert creator._create_datapipe("train") == ["a", "b", "c"]
    data_creator.create.assert_called_once_with()
    assert creator._data == {"train": [1, 2, 3]}
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=([1, 2, 3],))


def test_data_creator_iter_data_pipe_creator_datasource_create_datapipe_no_data_creator() -> None:
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train")
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_iter_data_pipe_creator_datasource_create_datapipe_no_data_creator_with_engine() -> (
    None
):
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train", engine)
    datapipe_creator.create.assert_called_once_with(engine=engine, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_iter_data_pipe_creator_datasource_get_dataloader() -> None:
    creator = DataCreatorIterDataPipeCreatorDataSource(
        datapipe_creators={
            "train": SequentialDataPipeCreator(
                config=[
                    {OBJECT_TARGET: "gravitorch.datapipes.iter.SourceWrapper"},
                ]
            )
        },
        data_creators={
            "train": HypercubeVertexDataCreator(num_examples=10, num_classes=5),
        },
    )
    datapipe = creator.get_dataloader("train")
    assert isinstance(datapipe, SourceWrapper)
