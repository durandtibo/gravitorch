import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch.nn import Module

from gravitorch.creators.core import (
    BaseCoreCreator,
    is_core_creator_config,
    setup_core_creator,
)

############################################
#     Tests for is_core_creator_config     #
############################################


def test_is_core_creator_config_true() -> None:
    assert is_core_creator_config(
        {
            OBJECT_TARGET: "gravitorch.creators.core.CoreCreator",
            "datasource": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
            "model": {OBJECT_TARGET: "gravitorch.testing.DummyClassificationModel"},
            "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
            "lr_scheduler": {OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        }
    )


def test_is_core_creator_config_false() -> None:
    assert not is_core_creator_config({OBJECT_TARGET: "torch.nn.Identity"})


########################################
#     Tests for setup_core_creator     #
########################################


def test_setup_core_creator_object() -> None:
    creator = Mock(spec=BaseCoreCreator)
    assert setup_core_creator(creator) is creator


def test_setup_core_creator_dict() -> None:
    assert isinstance(
        setup_core_creator(
            {
                OBJECT_TARGET: "gravitorch.creators.core.CoreCreator",
                "datasource": {OBJECT_TARGET: "gravitorch.testing.DummyDataSource"},
                "model": {OBJECT_TARGET: "gravitorch.testing.DummyClassificationModel"},
                "optimizer": {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
                "lr_scheduler": {OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
            }
        ),
        BaseCoreCreator,
    )


def test_setup_core_creator_dict_mock() -> None:
    factory_mock = Mock(return_value=Mock(spec=BaseCoreCreator))
    with patch("gravitorch.creators.core.base.BaseCoreCreator.factory", factory_mock):
        assert isinstance(setup_core_creator({OBJECT_TARGET: "name"}), BaseCoreCreator)
        factory_mock.assert_called_once_with(_target_="name")


def test_setup_core_creator_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_core_creator({OBJECT_TARGET: "torch.nn.Identity"}), Module)
        assert caplog.messages
