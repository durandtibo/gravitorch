from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.datasource import (
    BaseDataSourceCreator,
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


def test_setup_datasource_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.datasource.base.BaseDataSourceCreator", creator_mock):
        assert setup_datasource_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
