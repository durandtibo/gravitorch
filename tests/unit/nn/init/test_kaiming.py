import logging
import math
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.nn import freeze_module
from gravitorch.nn.init import (
    KaimingNormal,
    KaimingUniform,
    constant,
    kaiming_normal,
    kaiming_uniform,
)

###################################
#     Tests for KaimingNormal     #
###################################


def test_kaiming_normal_str() -> None:
    assert str(KaimingNormal()).startswith("KaimingNormal(")


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_normal_neg_slope(neg_slope: float) -> None:
    assert KaimingNormal(neg_slope=neg_slope)._neg_slope == neg_slope


def test_kaiming_normal_neg_slope_default() -> None:
    assert KaimingNormal()._neg_slope == 0.0


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_normal_mode(mode: str) -> None:
    assert KaimingNormal(mode=mode)._mode == mode


def test_kaiming_normal_default() -> None:
    assert KaimingNormal()._mode == "fan_in"


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_normal_nonlinearity(nonlinearity: str) -> None:
    assert KaimingNormal(nonlinearity=nonlinearity)._nonlinearity == nonlinearity


def test_kaiming_normal_nonlinearity_default() -> None:
    assert KaimingNormal()._nonlinearity == "leaky_relu"


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_normal_learnable_only(learnable_only: bool) -> None:
    assert KaimingNormal(learnable_only=learnable_only)._learnable_only == learnable_only


def test_kaiming_normal_learnable_only_default() -> None:
    assert KaimingNormal()._learnable_only


def test_kaiming_normal_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 1)
    KaimingNormal().initialize(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_kaiming_normal_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant(module, 1)
    KaimingNormal().initialize(module)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.ones(6))
    assert not module[2].weight.data.equal(torch.zeros(6, 6))
    assert module[2].bias.data.equal(torch.ones(6))


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_normal_initialize_neg_slope(neg_slope: float) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_normal") as kaiming:
        KaimingNormal(neg_slope=neg_slope).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=neg_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_normal_initialize_mode(mode: str) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_normal") as kaiming:
        KaimingNormal(mode=mode).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode=mode,
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_normal_initialize_nonlinearity(nonlinearity: str) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_normal") as kaiming:
        KaimingNormal(nonlinearity=nonlinearity).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity=nonlinearity,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_normal_initialize_learnable_only(
    learnable_only: bool,
) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_normal") as kaiming:
        KaimingNormal(learnable_only=learnable_only).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=learnable_only,
            log_info=False,
        )


@mark.parametrize("log_info", (True, False))
def test_kaiming_normal_initialize_log_info(
    log_info: bool,
) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_normal") as kaiming:
        KaimingNormal(log_info=log_info).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=log_info,
        )


####################################
#     Tests for KaimingUniform     #
####################################


def test_kaiming_uniform_str() -> None:
    assert str(KaimingUniform()).startswith("KaimingUniform(")


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_uniform_neg_slope(neg_slope: float) -> None:
    assert KaimingUniform(neg_slope=neg_slope)._neg_slope == neg_slope


def test_kaiming_uniform_neg_slope_default() -> None:
    assert KaimingUniform()._neg_slope == 0.0


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_uniform_mode(mode: str) -> None:
    assert KaimingUniform(mode=mode)._mode == mode


def test_kaiming_uniform_default() -> None:
    assert KaimingUniform()._mode == "fan_in"


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_uniform_nonlinearity(nonlinearity: str) -> None:
    assert KaimingUniform(nonlinearity=nonlinearity)._nonlinearity == nonlinearity


def test_kaiming_uniform_nonlinearity_default() -> None:
    assert KaimingUniform()._nonlinearity == "leaky_relu"


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_uniform_learnable_only(learnable_only: bool) -> None:
    assert KaimingUniform(learnable_only=learnable_only)._learnable_only == learnable_only


def test_kaiming_uniform_learnable_only_default() -> None:
    assert KaimingUniform()._learnable_only


def test_kaiming_uniform_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 1)
    KaimingUniform().initialize(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_kaiming_uniform_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant(module, 1)
    KaimingUniform().initialize(module)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.ones(6))
    assert not module[2].weight.data.equal(torch.zeros(6, 6))
    assert module[2].bias.data.equal(torch.ones(6))


@mark.parametrize("neg_slope", (1, 2.0))
def test_kaiming_uniform_initialize_neg_slope(neg_slope: float) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_uniform") as kaiming:
        KaimingUniform(neg_slope=neg_slope).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=neg_slope,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("mode", ("fan_in", "fan_out"))
def test_kaiming_uniform_initialize_mode(mode: str) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_uniform") as kaiming:
        KaimingUniform(mode=mode).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode=mode,
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("nonlinearity", ("relu", "leaky_relu"))
def test_kaiming_uniform_initialize_nonlinearity(nonlinearity: str) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_uniform") as kaiming:
        KaimingUniform(nonlinearity=nonlinearity).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity=nonlinearity,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("learnable_only", (True, False))
def test_kaiming_uniform_initialize_learnable_only(
    learnable_only: bool,
) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_uniform") as kaiming:
        KaimingUniform(learnable_only=learnable_only).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=learnable_only,
            log_info=False,
        )


@mark.parametrize("log_info", (True, False))
def test_kaiming_uniform_initialize_log_info(
    log_info: bool,
) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.kaiming.kaiming_uniform") as kaiming:
        KaimingUniform(log_info=log_info).initialize(module)
        kaiming.assert_called_once_with(
            module=module,
            neg_slope=0.0,
            mode="fan_in",
            nonlinearity="leaky_relu",
            learnable_only=True,
            log_info=log_info,
        )


####################################
#     Tests for kaiming_normal     #
####################################


def test_kaiming_normal_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    kaiming_normal(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Kaiming Normal does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


def test_kaiming_normal_nonlinearity_mode_fan_in() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_normal(module, mode="fan_in")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_kaiming_normal_nonlinearity_mode_fan_out() -> None:
    module = nn.Linear(10, 1000)
    constant(module, 0)
    kaiming_normal(module, mode="fan_out")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.044721359549995794, rel_tol=0.02
    )  # 2% tolerance


def test_kaiming_normal_nonlinearity_relu() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_normal(module, nonlinearity="relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_kaiming_normal_nonlinearity_leaky_relu() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_normal(module, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(
        module.weight.data.std().item(), 0.1414213562373095, rel_tol=0.02
    )  # 2% tolerance


def test_kaiming_normal_nonlinearity_leaky_relu_neg_slope_1() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_normal(module, neg_slope=-1, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert math.isclose(module.weight.data.std().item(), 0.1, rel_tol=0.02)  # 2% tolerance


def test_kaiming_normal_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    kaiming_normal(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_kaiming_normal_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    kaiming_normal(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_kaiming_normal_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        kaiming_normal(module, log_info=True)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert caplog.messages


def test_kaiming_normal_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        kaiming_normal(module)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert not caplog.messages


#####################################
#     Tests for kaiming_uniform     #
#####################################


def test_kaiming_uniform_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    kaiming_uniform(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    # The bias should not be initialized because Kaiming uniform does not work on vectors
    assert module.bias.data.equal(torch.zeros(6))


def test_kaiming_uniform_nonlinearity_mode_fan_in() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_uniform(module, mode="fan_in")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_kaiming_uniform_nonlinearity_mode_fan_out() -> None:
    module = nn.Linear(10, 1000)
    constant(module, 0)
    kaiming_uniform(module, mode="fan_out")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.07745966692414834
    assert module.weight.data.max().item() <= 0.07745966692414834


def test_kaiming_uniform_nonlinearity_relu() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_uniform(module, nonlinearity="relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_kaiming_uniform_nonlinearity_leaky_relu() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_uniform(module, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.2449489742783178
    assert module.weight.data.max().item() <= 0.2449489742783178


def test_kaiming_uniform_nonlinearity_leaky_relu_neg_slope_1() -> None:
    module = nn.Linear(100, 100)
    constant(module, 0)
    kaiming_uniform(module, neg_slope=-1, nonlinearity="leaky_relu")
    assert math.isclose(module.weight.data.mean().item(), 0.0, abs_tol=0.01)
    assert module.weight.data.min().item() >= -0.17320508075688773
    assert module.weight.data.max().item() <= 0.17320508075688773


def test_kaiming_uniform_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    kaiming_uniform(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_kaiming_uniform_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    kaiming_uniform(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_kaiming_uniform_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        kaiming_uniform(module, log_info=True)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert caplog.messages


def test_kaiming_uniform_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        kaiming_uniform(module)
        assert not module.weight.data.equal(torch.zeros(6, 4))
        assert module.bias.data.equal(torch.zeros(6))
        assert not caplog.messages
