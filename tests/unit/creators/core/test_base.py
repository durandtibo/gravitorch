from unittest.mock import Mock, patch

from gravitorch.creators.core import BaseCoreCreator, setup_core_creator

########################################
#     Tests for setup_core_creator     #
########################################


def test_setup_core_creator_object() -> None:
    creator = Mock(spec=BaseCoreCreator)
    assert setup_core_creator(creator) is creator


def test_setup_core_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.core.base.BaseCoreCreator", creator_mock):
        assert setup_core_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
