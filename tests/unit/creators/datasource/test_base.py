from unittest.mock import Mock, patch

from gravitorch.creators.datasource import (
    BaseDataSourceCreator,
    setup_datasource_creator,
)

###############################################
#     Tests for setup_datasource_creator     #
###############################################


def test_setup_datasource_creator_object() -> None:
    creator = Mock(spec=BaseDataSourceCreator)
    assert setup_datasource_creator(creator) is creator


def test_setup_datasource_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.datasource.base.BaseDataSourceCreator", creator_mock):
        assert setup_datasource_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
