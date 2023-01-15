from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.optimizer import (
    NoOptimizerCreator,
    VanillaOptimizerCreator,
    setup_optimizer_creator,
)

#############################################
#     Tests for setup_optimizer_creator     #
#############################################


def test_setup_optimizer_creator_none():
    assert isinstance(setup_optimizer_creator(None), NoOptimizerCreator)


def test_setup_optimizer_creator_object():
    optimizer_creator = VanillaOptimizerCreator()
    assert setup_optimizer_creator(optimizer_creator) is optimizer_creator


def test_setup_optimizer_creator_dict():
    assert isinstance(
        setup_optimizer_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.optimizer.VanillaOptimizerCreator",
            }
        ),
        VanillaOptimizerCreator,
    )


def test_setup_optimizer_creator_dict_mock():
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.optimizer.utils.BaseOptimizerCreator", creator_mock):
        assert setup_optimizer_creator({"_target_": "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
