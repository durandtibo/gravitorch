from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.lr_scheduler import (
    LRSchedulerCreator,
    is_lr_scheduler_creator_config,
    setup_lr_scheduler_creator,
)

####################################################
#     Tests for is_lr_scheduler_creator_config     #
####################################################


def test_is_lr_scheduler_creator_config_true() -> None:
    assert is_lr_scheduler_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.lr_scheduler.LRSchedulerCreator",
            "lr_scheduler_config": {
                OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR",
                "step_size": 5,
            },
        }
    )


def test_is_lr_scheduler_creator_config_false() -> None:
    assert not is_lr_scheduler_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


################################################
#     Tests for setup_lr_scheduler_creator     #
################################################


def test_setup_lr_scheduler_creator_none() -> None:
    assert isinstance(setup_lr_scheduler_creator(None), LRSchedulerCreator)


def test_setup_lr_scheduler_creator_object() -> None:
    lr_scheduler_creator = LRSchedulerCreator()
    assert setup_lr_scheduler_creator(lr_scheduler_creator) is lr_scheduler_creator


def test_setup_lr_scheduler_creator_dict() -> None:
    assert isinstance(
        setup_lr_scheduler_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.lr_scheduler.LRSchedulerCreator",
            }
        ),
        LRSchedulerCreator,
    )


def test_setup_lr_scheduler_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.lr_scheduler.factory.BaseLRSchedulerCreator", creator_mock):
        assert setup_lr_scheduler_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
