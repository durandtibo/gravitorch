from __future__ import annotations

from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.data.datacreators import (
    BaseDataCreator,
    HypercubeVertexDataCreator,
    is_datacreator_config,
    setup_data_creator,
)

###########################################
#     Tests for is_datacreator_config     #
###########################################


def test_is_datacreator_config_true() -> None:
    assert is_datacreator_config(
        {OBJECT_TARGET: "gravitorch.data.datacreators.HypercubeVertexDataCreator"}
    )


def test_is_datacreator_config_false() -> None:
    assert not is_datacreator_config({OBJECT_TARGET: "torch.nn.Identity"})


########################################
#     Tests for setup_data_creator     #
########################################


def test_setup_data_creator_object() -> None:
    creator = Mock(spec=BaseDataCreator)
    assert setup_data_creator(creator) is creator


def test_setup_data_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.data.datacreators.base.BaseDataCreator", creator_mock):
        assert setup_data_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")


def test_setup_data_creator_dict() -> None:
    assert isinstance(
        setup_data_creator(
            {OBJECT_TARGET: "gravitorch.data.datacreators.HypercubeVertexDataCreator"}
        ),
        HypercubeVertexDataCreator,
    )
