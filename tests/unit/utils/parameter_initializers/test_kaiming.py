import math
from unittest.mock import Mock, patch

import torch
from pytest import mark
from torch import nn

from gravitorch.engines import BaseEngine
from gravitorch.nn import freeze_module
from gravitorch.utils.parameter_initializers import (
    KaimingNormalParameterInitializer,
    KaimingUniformParameterInitializer,
    recursive_constant_,
    recursive_kaiming_normal_,
    recursive_kaiming_uniform_,
)

#######################################################
#     Tests for KaimingNormalParameterInitializer     #
#######################################################


def test_kaiming_normal_parameter_initializer_str():
    assert str(KaimingNormalParameterInitializer()).startswith("KaimingNormalParameterInitializer(")


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_normal_parameter_initializer_neg_slope(neg_slope: float):
    assert KaimingNormalParameterInitializer(neg_slope=neg_slope)._neg_slope == neg_slope


def test_kaiming_normal_parameter_initializer_neg_slope_default():
    assert KaimingNormalParameterInitializer()._neg_slope == 0.0


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_normal_parameter_initializer_mode(mode: str):
    assert KaimingNormalParameterInitializer(mode=mode)._mode == mode


def test_kaiming_normal_parameter_initializer_default():
    assert KaimingNormalParameterInitializer()._mode == "fan_in"


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_normal_parameter_initializer_nonlinearity(nonlinearity: str):
    assert (
        KaimingNormalParameterInitializer(nonlinearity=nonlinearity)._nonlinearity == nonlinearity
    )


def test_kaiming_normal_parameter_initializer_nonlinearity_default():
    assert KaimingNormalParameterInitializer()._nonlinearity == "leaky_relu"


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_normal_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        KaimingNormalParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_kaiming_normal_parameter_initializer_learnable_only_default():
    assert KaimingNormalParameterInitializer()._learnable_only


@mark.parametrize("show_stats", (True, False))
def test_kaiming_normal_parameter_initializer_show_stats(show_stats: bool):
    assert KaimingNormalParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_kaiming_normal_parameter_initializer_show_stats_default():
    assert KaimingNormalParameterInitializer()._show_stats


def test_kaiming_normal_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    KaimingNormalParameterInitializer().initialize(engine)
    assert not engine.model.weight.data.equal(torch.zeros(6, 4))
    assert engine.model.bias.data.equal(torch.ones(6))


def test_kaiming_normal_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 1)
    KaimingNormalParameterInitializer().initialize(engine)
    assert not engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert engine.model[0].bias.data.equal(torch.ones(6))
    assert not engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert engine.model[2].bias.data.equal(torch.ones(6))


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_normal_parameter_initializer_initialize_neg_slope(neg_slope: float):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_normal_"
    ) as kaiming:
        KaimingNormalParameterInitializer(neg_slope=neg_slope).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=neg_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
        )


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_normal_parameter_initializer_initialize_mode(mode: str):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_normal_"
    ) as kaiming:
        KaimingNormalParameterInitializer(mode=mode).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode=mode,
            nonlinearity="leaky_relu",
            learnable_only=True,
        )


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_normal_parameter_initializer_initialize_nonlinearity(nonlinearity: str):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_normal_"
    ) as kaiming:
        KaimingNormalParameterInitializer(nonlinearity=nonlinearity).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity=nonlinearity,
            learnable_only=True,
        )


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_normal_parameter_initializer_initialize_learnable_only(learnable_only: bool):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_normal_"
    ) as kaiming:
        KaimingNormalParameterInitializer(learnable_only=learnable_only).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=learnable_only,
        )


#######################################################
#     Tests for KaimingUniformParameterInitializer     #
#######################################################


def test_kaiming_uniform_parameter_initializer_str():
    assert str(KaimingUniformParameterInitializer()).startswith(
        "KaimingUniformParameterInitializer("
    )


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_uniform_parameter_initializer_neg_slope(neg_slope: float):
    assert KaimingUniformParameterInitializer(neg_slope=neg_slope)._neg_slope == neg_slope


def test_kaiming_uniform_parameter_initializer_neg_slope_default():
    assert KaimingUniformParameterInitializer()._neg_slope == 0.0


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_uniform_parameter_initializer_mode(mode: str):
    assert KaimingUniformParameterInitializer(mode=mode)._mode == mode


def test_kaiming_uniform_parameter_initializer_default():
    assert KaimingUniformParameterInitializer()._mode == "fan_in"


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_uniform_parameter_initializer_nonlinearity(nonlinearity: str):
    assert (
        KaimingUniformParameterInitializer(nonlinearity=nonlinearity)._nonlinearity == nonlinearity
    )


def test_kaiming_uniform_parameter_initializer_nonlinearity_default():
    assert KaimingUniformParameterInitializer()._nonlinearity == "leaky_relu"


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_uniform_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        KaimingUniformParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_kaiming_uniform_parameter_initializer_learnable_only_default():
    assert KaimingUniformParameterInitializer()._learnable_only


@mark.parametrize("show_stats", (True, False))
def test_kaiming_uniform_parameter_initializer_show_stats(show_stats: bool):
    assert KaimingUniformParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_kaiming_uniform_parameter_initializer_show_stats_default():
    assert KaimingUniformParameterInitializer()._show_stats


def test_kaiming_uniform_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    KaimingUniformParameterInitializer().initialize(engine)
    assert not engine.model.weight.data.equal(torch.zeros(6, 4))
    assert engine.model.bias.data.equal(torch.ones(6))


def test_kaiming_uniform_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 1)
    KaimingUniformParameterInitializer().initialize(engine)
    assert not engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert engine.model[0].bias.data.equal(torch.ones(6))
    assert not engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert engine.model[2].bias.data.equal(torch.ones(6))


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_uniform_parameter_initializer_initialize_neg_slope(neg_slope: float):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_uniform_"
    ) as kaiming:
        KaimingUniformParameterInitializer(neg_slope=neg_slope).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=neg_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
        )


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_uniform_parameter_initializer_initialize_mode(mode: str):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_uniform_"
    ) as kaiming:
        KaimingUniformParameterInitializer(mode=mode).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode=mode,
            nonlinearity="leaky_relu",
            learnable_only=True,
        )


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_uniform_parameter_initializer_initialize_nonlinearity(nonlinearity: str):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_uniform_"
    ) as kaiming:
        KaimingUniformParameterInitializer(nonlinearity=nonlinearity).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity=nonlinearity,
            learnable_only=True,
        )


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_uniform_parameter_initializer_initialize_learnable_only(learnable_only: bool):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.kaiming.recursive_kaiming_uniform_"
    ) as kaiming:
        KaimingUniformParameterInitializer(learnable_only=learnable_only).initialize(engine)
        kaiming.assert_called_once_with(
            module=engine.model,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=learnable_only,
        )


###############################################
#     Tests for recursive_kaiming_normal_     #
###############################################


def test_recursive_kaiming_normal_linear():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Kaiming Normal does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


def test_recursive_kaiming_normal_nonlinearity_mode_fan_in():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module, mode="fan_in")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_recursive_kaiming_normal_nonlinearity_mode_fan_out():
    module = nn.Linear(10, 1000)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module, mode="fan_out")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.044721359549995794, rel_tol=0.02
    )  # 2% tolerance


def test_recursive_kaiming_normal_nonlinearity_relu():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module, nonlinearity="relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_recursive_kaiming_normal_nonlinearity_leaky_relu():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_recursive_kaiming_normal_nonlinearity_leaky_relu_neg_slope_1():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_normal_(module, neg_slope=-1, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(module.weight.data.std().item(), 0.1, rel_tol=0.02)  # 2% tolerance


def test_recursive_kaiming_normal_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_kaiming_normal_(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_kaiming_normal_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_kaiming_normal_(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


################################################
#     Tests for recursive_kaiming_uniform_     #
################################################


def test_recursive_kaiming_uniform_linear():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Kaiming uniform does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


def test_recursive_kaiming_uniform_nonlinearity_mode_fan_in():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module, mode="fan_in")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_recursive_kaiming_uniform_nonlinearity_mode_fan_out():
    module = nn.Linear(10, 1000)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module, mode="fan_out")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.07745966692414834
    assert module.weight.data.max().item() <= 0.07745966692414834


def test_recursive_kaiming_uniform_nonlinearity_relu():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module, nonlinearity="relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_recursive_kaiming_uniform_nonlinearity_leaky_relu():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_recursive_kaiming_uniform_nonlinearity_leaky_relu_neg_slope_1():
    module = nn.Linear(100, 100)
    recursive_constant_(module, 0)
    recursive_kaiming_uniform_(module, neg_slope=-1, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.17320508075688773
    assert module.weight.data.max().item() <= 0.17320508075688773


def test_recursive_kaiming_uniform_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_kaiming_uniform_(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_kaiming_uniform_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_kaiming_uniform_(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))
