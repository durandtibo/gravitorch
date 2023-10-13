import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises
from torch.utils.data.datapipes.iter import IterableWrapper

from gravitorch.creators.datapipe import BaseDataPipeCreator, ChainedDataPipeCreator
from gravitorch.data.datacreators import BaseDataCreator, HypercubeVertexDataCreator
from gravitorch.datasources import (
    DataCreatorDataSource,
    DataPipeDataSource,
    DataStreamNotFoundError,
)
from gravitorch.engines import BaseEngine
from gravitorch.utils.asset import AssetNotFoundError

########################################
#     Tests for DataPipeDataSource     #
########################################


@fixture
def datasource() -> DataPipeDataSource:
    return DataPipeDataSource(
        {
            "train": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.ChainedDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                        "iterable": [1, 2, 3, 4],
                    },
                ],
            },
            "val": {
                OBJECT_TARGET: "gravitorch.creators.datapipe.ChainedDataPipeCreator",
                "config": [
                    {
                        OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                        "iterable": ["a", "b", "c"],
                    },
                ],
            },
        }
    )


def test_datapipe_datasource_str() -> None:
    assert str(DataPipeDataSource({"train": Mock(spec=BaseDataPipeCreator)})).startswith(
        "DataPipeDataSource("
    )


def test_datapipe_datasource_attach(
    caplog: LogCaptureFixture, datasource: DataPipeDataSource
) -> None:
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_datapipe_datasource_get_asset_exists(
    datasource: DataPipeDataSource,
) -> None:
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_datapipe_datasource_get_asset_does_not_exist(
    datasource: DataPipeDataSource,
) -> None:
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_datapipe_datasource_has_asset_true(
    datasource: DataPipeDataSource,
) -> None:
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_datapipe_datasource_has_asset_false(
    datasource: DataPipeDataSource,
) -> None:
    assert not datasource.has_asset("something")


def test_datapipe_datasource_get_datastream_train(
    datasource: DataPipeDataSource,
) -> None:
    loader = datasource.get_datastream("train")
    assert isinstance(loader, IterableWrapper)
    assert tuple(loader) == (1, 2, 3, 4)


def test_datapipe_datasource_get_datastream_val(
    datasource: DataPipeDataSource,
) -> None:
    loader = datasource.get_datastream("val")
    assert isinstance(loader, IterableWrapper)
    assert tuple(loader) == ("a", "b", "c")


def test_datapipe_datasource_get_datastream_missing(
    datasource: DataPipeDataSource,
) -> None:
    with raises(DataStreamNotFoundError):
        datasource.get_datastream("missing")


def test_datapipe_datasource_get_datastream_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = DataPipeDataSource({"train": datapipe_creator})
    datasource.get_datastream("train", engine=engine)
    datapipe_creator.create.assert_called_once_with(engine=engine)


def test_datapipe_datasource_get_datastream_without_engine() -> None:
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = DataPipeDataSource({"train": datapipe_creator})
    datasource.get_datastream("train")
    datapipe_creator.create.assert_called_once_with(engine=None)


def test_datapipe_datasource_has_datastream_true(
    datasource: DataPipeDataSource,
) -> None:
    assert datasource.has_datastream("train")


def test_datapipe_datasource_has_datastream_false(
    datasource: DataPipeDataSource,
) -> None:
    assert not datasource.has_datastream("missing")


def test_datapipe_datasource_load_state_dict(
    datasource: DataPipeDataSource,
) -> None:
    datasource.load_state_dict({})


def test_datapipe_datasource_state_dict(
    datasource: DataPipeDataSource,
) -> None:
    assert datasource.state_dict() == {}


###########################################
#     Tests for DataCreatorDataSource     #
###########################################


def test_data_creator_datasource_str() -> None:
    assert str(
        DataCreatorDataSource(
            datapipe_creators={"train": Mock(spec=BaseDataPipeCreator)},
            data_creators={"train": Mock(spec=BaseDataCreator)},
        )
    ).startswith("DataCreatorDataSource(")


def test_data_creator_datasource_data_creators() -> None:
    creator = DataCreatorDataSource(
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


def test_data_creator_datasource_create_datapipe() -> None:
    data_creator = Mock(spec=BaseDataCreator, create=Mock(return_value=[1, 2, 3]))
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={"train": data_creator},
    )
    assert creator._create_datapipe("train") == ["a", "b", "c"]
    data_creator.create.assert_called_once_with()
    assert creator._data == {"train": [1, 2, 3]}
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=([1, 2, 3],))


def test_data_creator_datasource_create_datapipe_no_data_creator() -> None:
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train")
    datapipe_creator.create.assert_called_once_with(engine=None, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_datasource_create_datapipe_no_data_creator_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    datapipe_creator = Mock(spec=BaseDataPipeCreator, create=Mock(return_value=["a", "b", "c"]))
    creator = DataCreatorDataSource(
        datapipe_creators={"train": datapipe_creator},
        data_creators={},
    )
    datapipe = creator._create_datapipe("train", engine)
    datapipe_creator.create.assert_called_once_with(engine=engine, source_inputs=None)
    assert tuple(datapipe) == ("a", "b", "c")


def test_data_creator_datasource_get_datastream() -> None:
    creator = DataCreatorDataSource(
        datapipe_creators={
            "train": ChainedDataPipeCreator(
                config=[
                    {OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper"},
                ]
            )
        },
        data_creators={
            "train": HypercubeVertexDataCreator(num_examples=10, num_classes=5),
        },
    )
    datapipe = creator.get_datastream("train")
    assert isinstance(datapipe, IterableWrapper)
