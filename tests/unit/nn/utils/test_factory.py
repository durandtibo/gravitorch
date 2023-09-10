from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark
from torch import nn
from torch.nn import ReLU

from gravitorch.engines import BaseEngine
from gravitorch.handlers import EpochLRMonitor
from gravitorch.nn import is_module_config, setup_module
from gravitorch.nn.utils.factory import attach_module_to_engine

#############################################
#     Tests for attach_module_to_engine     #
#############################################


def test_attach_module_to_engine_attach() -> None:
    engine = Mock(spec=BaseEngine)
    module = Mock()
    attach_module_to_engine(module, engine)
    module.attach.assert_called_once_with(engine)


def test_attach_module_to_engine_no_attach() -> None:
    engine = Mock(spec=BaseEngine)
    module = Mock()
    with patch("gravitorch.nn.utils.factory.hasattr", lambda *args, **kwargs: False):
        attach_module_to_engine(module, engine)
        module.attach.assert_not_called()


######################################
#     Tests for is_module_config     #
######################################


def test_is_module_config_true() -> None:
    assert is_module_config({OBJECT_TARGET: "torch.nn.Identity"})


def test_is_module_config_false() -> None:
    assert not is_module_config({OBJECT_TARGET: "gravitorch.loops.training.TrainingLoop"})


##################################
#     Tests for setup_module     #
##################################


@mark.parametrize("module", (ReLU(), {OBJECT_TARGET: "torch.nn.ReLU"}))
def test_setup_module(module: nn.Module | dict) -> None:
    assert isinstance(setup_module(module), ReLU)


def test_setup_module_object() -> None:
    module = ReLU()
    assert setup_module(module) is module


def test_setup_module_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(
            setup_module({OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"}), EpochLRMonitor
        )
        assert caplog.messages
