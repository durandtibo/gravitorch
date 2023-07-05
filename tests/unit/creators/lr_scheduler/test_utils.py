from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.creators.lr_scheduler import (
    VanillaLRSchedulerCreator,
    setup_lr_scheduler_creator,
)

################################################
#     Tests for setup_lr_scheduler_creator     #
################################################


def test_setup_lr_scheduler_creator_none() -> None:
    assert isinstance(setup_lr_scheduler_creator(None), VanillaLRSchedulerCreator)


def test_setup_lr_scheduler_creator_object() -> None:
    lr_scheduler_creator = VanillaLRSchedulerCreator()
    assert setup_lr_scheduler_creator(lr_scheduler_creator) is lr_scheduler_creator


def test_setup_lr_scheduler_creator_dict() -> None:
    assert isinstance(
        setup_lr_scheduler_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.lr_scheduler.VanillaLRSchedulerCreator",
            }
        ),
        VanillaLRSchedulerCreator,
    )


def test_setup_lr_scheduler_creator_dict_mock() -> None:
    creator_mock = Mock(factory=Mock(return_value="abc"))
    with patch("gravitorch.creators.lr_scheduler.utils.BaseLRSchedulerCreator", creator_mock):
        assert setup_lr_scheduler_creator({OBJECT_TARGET: "name"}) == "abc"
        creator_mock.factory.assert_called_once_with(_target_="name")
