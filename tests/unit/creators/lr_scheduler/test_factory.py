import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.lr_scheduler import (
    BaseLRSchedulerCreator,
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
    factory_mock = Mock(return_value=Mock(spec=BaseLRSchedulerCreator))
    with patch(
        "gravitorch.creators.lr_scheduler.factory.BaseLRSchedulerCreator.factory", factory_mock
    ):
        assert isinstance(
            setup_lr_scheduler_creator({OBJECT_TARGET: "name"}), BaseLRSchedulerCreator
        )
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_lr_scheduler_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_lr_scheduler_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
