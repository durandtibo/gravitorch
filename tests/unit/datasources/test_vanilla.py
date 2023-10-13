from __future__ import annotations

import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, raises

from gravitorch.creators.datastream import (
    BaseDataStreamCreator,
    IterableDataStreamCreator,
)
from gravitorch.datasources import IterableNotFoundError, VanillaDataSource
from gravitorch.datastreams import IterableDataStream
from gravitorch.engines import BaseEngine
from gravitorch.utils.asset import AssetNotFoundError

#######################################
#     Tests for VanillaDataSource     #
#######################################


@fixture
def datasource() -> VanillaDataSource:
    return VanillaDataSource(
        {
            "train": {
                OBJECT_TARGET: "gravitorch.creators.datastream.IterableDataStreamCreator",
                "iterable": [1, 2, 3, 4],
            },
            "eval": IterableDataStreamCreator(["a", "b", "c"]),
        }
    )


def test_vanilla_data_source_str(datasource: VanillaDataSource) -> None:
    assert str(datasource).startswith("VanillaDataSource(")


def test_vanilla_data_source_attach(
    caplog: LogCaptureFixture, datasource: VanillaDataSource
) -> None:
    with caplog.at_level(logging.INFO):
        datasource.attach(engine=Mock(spec=BaseEngine))
        assert len(caplog.messages) >= 1


def test_vanilla_data_source_get_asset_exists(datasource: VanillaDataSource) -> None:
    datasource._asset_manager.add_asset("something", 2)
    assert datasource.get_asset("something") == 2


def test_vanilla_data_source_get_asset_does_not_exist(datasource: VanillaDataSource) -> None:
    with raises(AssetNotFoundError, match="The asset 'something' does not exist"):
        datasource.get_asset("something")


def test_vanilla_data_source_has_asset_true(datasource: VanillaDataSource) -> None:
    datasource._asset_manager.add_asset("something", 1)
    assert datasource.has_asset("something")


def test_vanilla_data_source_has_asset_false(datasource: VanillaDataSource) -> None:
    assert not datasource.has_asset("something")


def test_vanilla_data_source_get_iterable_train(datasource: VanillaDataSource) -> None:
    datastream = datasource.get_iterable("train")
    assert isinstance(datastream, IterableDataStream)
    with datastream as flow:
        assert tuple(flow) == (1, 2, 3, 4)


def test_vanilla_data_source_get_iterable_eval(datasource: VanillaDataSource) -> None:
    datastream = datasource.get_iterable("eval")
    assert isinstance(datastream, IterableDataStream)
    with datastream as flow:
        assert tuple(flow) == ("a", "b", "c")


def test_vanilla_data_source_get_iterable_missing(datasource: VanillaDataSource) -> None:
    with raises(IterableNotFoundError):
        datasource.get_iterable("missing")


def test_vanilla_data_source_get_iterable_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    datastream_creator = Mock(spec=BaseDataStreamCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = VanillaDataSource({"train": datastream_creator})
    datasource.get_iterable("train", engine=engine)
    datastream_creator.create.assert_called_once_with(engine=engine)


def test_vanilla_data_source_get_iterable_without_engine() -> None:
    datastream_creator = Mock(spec=BaseDataStreamCreator, create=Mock(return_value=["a", "b", "c"]))
    datasource = VanillaDataSource({"train": datastream_creator})
    datasource.get_iterable("train")
    datastream_creator.create.assert_called_once_with(engine=None)


def test_vanilla_data_source_has_iterable_true(datasource: VanillaDataSource) -> None:
    assert datasource.has_iterable("train")


def test_vanilla_data_source_has_iterable_false(datasource: VanillaDataSource) -> None:
    assert not datasource.has_iterable("missing")


def test_vanilla_data_source_load_state_dict(datasource: VanillaDataSource) -> None:
    datasource.load_state_dict({})


def test_vanilla_data_source_state_dict(datasource: VanillaDataSource) -> None:
    assert datasource.state_dict() == {}
