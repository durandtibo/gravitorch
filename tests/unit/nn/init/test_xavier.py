import logging
import math
from typing import Union
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.nn import freeze_module
from gravitorch.nn.init import (
    XavierNormal,
    XavierUniform,
    constant_init,
    xavier_normal_init,
    xavier_uniform_init,
)

##################################
#     Tests for XavierNormal     #
##################################


def test_xavier_normal_str() -> None:
    assert str(XavierNormal()).startswith("XavierNormal(")


@mark.parametrize("gain", (1, 2.0))
def test_xavier_normal_gain(gain: Union[int, float]) -> None:
    assert XavierNormal(gain=gain)._gain == gain


def test_xavier_normal_gain_default() -> None:
    assert XavierNormal()._gain == 1.0


@mark.parametrize("learnable_only", (True, False))
def test_xavier_normal_learnable_only(learnable_only: bool) -> None:
    assert XavierNormal(learnable_only=learnable_only)._learnable_only == learnable_only


def test_xavier_normal_learnable_only_default() -> None:
    assert XavierNormal()._learnable_only


def test_xavier_normal_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    XavierNormal().initialize(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_xavier_normal_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant_init(module, 1.0)
    XavierNormal().initialize(module)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.ones(6))
    assert not module[2].weight.data.equal(torch.zeros(6, 6))
    assert module[2].bias.data.equal(torch.ones(6))


@mark.parametrize("gain", (1, 2.0))
@mark.parametrize("learnable_only", (True, False))
def test_xavier_normal_initialize(gain: Union[int, float], learnable_only: bool) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with patch("gravitorch.nn.init.xavier.xavier_normal_init") as xavier:
        XavierNormal(gain=gain, learnable_only=learnable_only).initialize(module)
        xavier.assert_called_once_with(
            module=module, gain=gain, learnable_only=learnable_only, log_info=False
        )


###################################
#     Tests for XavierUniform     #
###################################


def test_xavier_uniform_str() -> None:
    assert str(XavierUniform()).startswith("XavierUniform(")


@mark.parametrize("gain", (1, 2.0))
def test_xavier_uniform_gain(gain: Union[int, float]) -> None:
    assert XavierUniform(gain=gain)._gain == gain


def test_xavier_uniform_gain_default() -> None:
    assert XavierUniform()._gain == 1.0


@mark.parametrize("learnable_only", (True, False))
def test_xavier_uniform_learnable_only(learnable_only: bool) -> None:
    assert XavierUniform(learnable_only=learnable_only)._learnable_only == learnable_only


def test_xavier_uniform_learnable_only_default() -> None:
    assert XavierUniform()._learnable_only


def test_xavier_uniform_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    XavierUniform().initialize(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_xavier_uniform_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant_init(module, 1.0)
    XavierUniform().initialize(module)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.ones(6))
    assert not module[2].weight.data.equal(torch.zeros(6, 6))
    assert module[2].bias.data.equal(torch.ones(6))


@mark.parametrize("gain", (1, 2.0))
@mark.parametrize("learnable_only", (True, False))
def test_xavier_uniform_initialize(gain: Union[int, float], learnable_only: bool) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with patch("gravitorch.nn.init.xavier.xavier_uniform_init") as xavier:
        XavierUniform(gain=gain, learnable_only=learnable_only).initialize(module)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.ones(6))
        xavier.assert_called_once_with(
            module=module, gain=gain, learnable_only=learnable_only, log_info=False
        )


########################################
#     Tests for xavier_normal_init     #
########################################


def test_xavier_normal_init_linear() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    xavier_normal_init(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Xavier Normal does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


@mark.parametrize("gain", (1.0, 2.0))
def test_xavier_normal_init_gain(gain: float) -> None:
    module = nn.Linear(100, 100)
    constant_init(module, 0.0)
    xavier_normal_init(module, gain=gain)
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(module.weight.data.std().item(), gain * 0.1, rel_tol=0.02)  # 2% tolerance


def test_xavier_normal_init_sequential_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 0.0)
    freeze_module(module[1])
    xavier_normal_init(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_xavier_normal_init_sequential_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 0.0)
    freeze_module(module[1])
    xavier_normal_init(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_xavier_normal_init_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        xavier_normal_init(module, log_info=True)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert caplog.messages


def test_xavier_normal_init_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        xavier_normal_init(module)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert not caplog.messages


#########################################
#     Tests for xavier_uniform_init     #
#########################################


def test_xavier_uniform_init_linear() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    xavier_uniform_init(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Xavier uniform does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


@mark.parametrize("gain", (0.1, 1.0, 2.0))
def test_xavier_uniform_init_gain(gain: float) -> None:
    module = nn.Linear(100, 100)
    constant_init(module, 0.0)
    xavier_uniform_init(module, gain=gain)
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.max().item() <= gain * 0.17320508075688773
    assert module.weight.data.min().item() >= -gain * 0.17320508075688773


def test_xavier_uniform_init_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 0.0)
    freeze_module(module[1])
    xavier_uniform_init(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_xavier_uniform_init_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 0.0)
    freeze_module(module[1])
    xavier_uniform_init(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_xavier_uniform_init_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        xavier_uniform_init(module, log_info=True)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert caplog.messages


def test_xavier_uniform_init_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        xavier_uniform_init(module)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert not caplog.messages
