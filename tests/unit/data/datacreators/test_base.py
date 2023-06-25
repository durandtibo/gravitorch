from __future__ import annotations

from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.data.datacreators import (
    BaseDataCreator,
    HypercubeVertexDataCreator,
    setup_data_creator,
)

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
