import math
from typing import Union
from unittest.mock import Mock, patch

import torch
from pytest import mark
from torch import nn

from gravitorch.engines import BaseEngine
from gravitorch.nn import freeze_module
from gravitorch.utils.parameter_initializers import (
    XavierNormalParameterInitializer,
    XavierUniformParameterInitializer,
    recursive_constant_,
    recursive_xavier_normal_,
    recursive_xavier_uniform_,
)

######################################################
#     Tests for XavierNormalParameterInitializer     #
######################################################


def test_xavier_normal_parameter_initializer_str():
    assert str(XavierNormalParameterInitializer()).startswith("XavierNormalParameterInitializer(")


@mark.parametrize("gain", (1, 2.0))
def test_xavier_normal_parameter_initializer_gain(gain: Union[int, float]):
    assert XavierNormalParameterInitializer(gain=gain)._gain == gain


def test_xavier_normal_parameter_initializer_gain_default():
    assert XavierNormalParameterInitializer()._gain == 1.0


@mark.parametrize("learnable_only", (True, False))
def test_xavier_normal_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        XavierNormalParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_xavier_normal_parameter_initializer_learnable_only_default():
    assert XavierNormalParameterInitializer()._learnable_only


@mark.parametrize("show_stats", (True, False))
def test_xavier_normal_parameter_initializer_show_stats(show_stats: bool):
    assert XavierNormalParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_xavier_normal_parameter_initializer_show_stats_default():
    assert XavierNormalParameterInitializer()._show_stats


def test_xavier_normal_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    XavierNormalParameterInitializer().initialize(engine)
    assert not engine.model.weight.data.equal(torch.zeros(6, 4))
    assert engine.model.bias.data.equal(torch.ones(6))


def test_xavier_normal_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 1)
    XavierNormalParameterInitializer().initialize(engine)
    assert not engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert engine.model[0].bias.data.equal(torch.ones(6))
    assert not engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert engine.model[2].bias.data.equal(torch.ones(6))


@mark.parametrize("gain", (1, 2.0))
@mark.parametrize("learnable_only", (True, False))
def test_xavier_normal_parameter_initializer_initialize(
    gain: Union[int, float], learnable_only: bool
):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch("gravitorch.utils.parameter_initializers.xavier.recursive_xavier_normal_") as xavier:
        XavierNormalParameterInitializer(gain=gain, learnable_only=learnable_only).initialize(
            engine
        )
        xavier.assert_called_once_with(
            module=engine.model, gain=gain, learnable_only=learnable_only
        )


#######################################################
#     Tests for XavierUniformParameterInitializer     #
#######################################################


def test_xavier_uniform_parameter_initializer_str():
    assert str(XavierUniformParameterInitializer()).startswith("XavierUniformParameterInitializer(")


@mark.parametrize("gain", (1, 2.0))
def test_xavier_uniform_parameter_initializer_gain(gain: Union[int, float]):
    assert XavierUniformParameterInitializer(gain=gain)._gain == gain


def test_xavier_uniform_parameter_initializer_gain_default():
    assert XavierUniformParameterInitializer()._gain == 1.0


@mark.parametrize("learnable_only", (True, False))
def test_xavier_uniform_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        XavierUniformParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_xavier_uniform_parameter_initializer_learnable_only_default():
    assert XavierUniformParameterInitializer()._learnable_only


@mark.parametrize("show_stats", (True, False))
def test_xavier_uniform_parameter_initializer_show_stats(show_stats: bool):
    assert XavierUniformParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_xavier_uniform_parameter_initializer_show_stats_default():
    assert XavierUniformParameterInitializer()._show_stats


def test_xavier_uniform_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    XavierUniformParameterInitializer().initialize(engine)
    assert not engine.model.weight.data.equal(torch.zeros(6, 4))
    assert engine.model.bias.data.equal(torch.ones(6))


def test_xavier_uniform_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 1)
    XavierUniformParameterInitializer().initialize(engine)
    assert not engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert engine.model[0].bias.data.equal(torch.ones(6))
    assert not engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert engine.model[2].bias.data.equal(torch.ones(6))


@mark.parametrize("gain", (1, 2.0))
@mark.parametrize("learnable_only", (True, False))
def test_xavier_uniform_parameter_initializer_initialize(
    gain: Union[int, float], learnable_only: bool
):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.xavier.recursive_xavier_uniform_"
    ) as xavier:
        XavierUniformParameterInitializer(gain=gain, learnable_only=learnable_only).initialize(
            engine
        )
        assert not engine.model.weight.data.equal(torch.zeros(6, 4))
        assert engine.model.bias.data.equal(torch.ones(6))
        xavier.assert_called_once_with(
            module=engine.model, gain=gain, learnable_only=learnable_only
        )


##############################################
#     Tests for recursive_xavier_normal_     #
##############################################


def test_recursive_xavier_normal_linear():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_xavier_normal_(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Xavier Normal does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


@mark.parametrize("gain", (1.0, 2.0))
def test_recursive_xavier_normal_gain(gain: float):
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_xavier_normal_(module, gain=gain)
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(module.weight.data.std().item(), gain * 0.1, rel_tol=0.02)  # 2% tolerance


def test_recursive_xavier_normal_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_xavier_normal_(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_xavier_normal_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_xavier_normal_(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


###############################################
#     Tests for recursive_xavier_uniform_     #
###############################################


def test_recursive_xavier_uniform_linear():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_xavier_uniform_(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Xavier uniform does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


@mark.parametrize("gain", (0.1, 1.0, 2.0))
def test_recursive_xavier_uniform_gain(gain: float):
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_xavier_uniform_(module, gain=gain)
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.max().item() <= gain * 0.17320508075688773
    assert module.weight.data.min().item() >= -gain * 0.17320508075688773


def test_recursive_xavier_uniform_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_xavier_uniform_(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_xavier_uniform_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_xavier_uniform_(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))
