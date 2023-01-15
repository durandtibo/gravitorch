from unittest.mock import Mock, patch

from gravitorch.creators.datasource import (
    BaseDataSourceCreator,
    setup_data_source_creator,
)

###############################################
#     Tests for setup_data_source_creator     #
###############################################


def test_setup_data_source_creator_object():
    creator = Mock(spec=BaseDataSourceCreator)
    assert setup_data_source_creator(creator) is creator


def test_setup_data_source_creator_dict_mock():
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.datasource.base.BaseDataSourceCreator", creator_mock):
        assert setup_data_source_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
