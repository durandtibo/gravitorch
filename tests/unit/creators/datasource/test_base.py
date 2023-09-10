import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.datasource import (
    BaseDataSourceCreator,
    DataSourceCreator,
    is_datasource_creator_config,
    setup_datasource_creator,
)

##################################################
#     Tests for is_datasource_creator_config     #
##################################################


def test_is_datasource_creator_config_true() -> None:
    assert is_datasource_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.datasource.DataSourceCreator",
            "config": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
        }
    )


def test_is_datasource_creator_config_false() -> None:
    assert not is_datasource_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


##############################################
#     Tests for setup_datasource_creator     #
##############################################


def test_setup_datasource_creator_object() -> None:
    creator = Mock(spec=BaseDataSourceCreator)
    assert setup_datasource_creator(creator) is creator


def test_setup_datasource_creator_dict() -> None:
    assert isinstance(
        setup_datasource_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.datasource.DataSourceCreator",
                "config": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
            }
        ),
        DataSourceCreator,
    )


def test_setup_datasource_creator_dict_mock() -> None:
    factory_mock = Mock(return_value=Mock(spec=BaseDataSourceCreator))
    with patch("gravitorch.creators.datasource.base.BaseDataSourceCreator.factory", factory_mock):
        assert isinstance(setup_datasource_creator({OBJECT_TARGET: "name"}), BaseDataSourceCreator)
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_datasource_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_datasource_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
