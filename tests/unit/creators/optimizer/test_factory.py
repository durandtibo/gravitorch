from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.optimizer import (
    NoOptimizerCreator,
    OptimizerCreator,
    is_optimizer_creator_config,
    setup_optimizer_creator,
)

#################################################
#     Tests for is_optimizer_creator_config     #
#################################################


def test_is_optimizer_creator_config_true() -> None:
    assert is_optimizer_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.optimizer.OptimizerCreator",
            "optimizer_config": {"_target_": "torch.optim.SGD", "lr": 0.01},
        }
    )


def test_is_optimizer_creator_config_false() -> None:
    assert not is_optimizer_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


#############################################
#     Tests for setup_optimizer_creator     #
#############################################


def test_setup_optimizer_creator_none() -> None:
    assert isinstance(setup_optimizer_creator(None), NoOptimizerCreator)


def test_setup_optimizer_creator_object() -> None:
    optimizer_creator = OptimizerCreator()
    assert setup_optimizer_creator(optimizer_creator) is optimizer_creator


def test_setup_optimizer_creator_dict() -> None:
    assert isinstance(
        setup_optimizer_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.optimizer.OptimizerCreator",
            }
        ),
        OptimizerCreator,
    )


def test_setup_optimizer_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.optimizer.factory.BaseOptimizerCreator", creator_mock):
        assert setup_optimizer_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
