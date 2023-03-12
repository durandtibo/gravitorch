import logging
from typing import Union
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.nn import freeze_module
from gravitorch.nn.init import ConstantBias, constant_bias_init, constant_init

MODULE_WITH_BIAS = (
    nn.Linear(4, 6),
    nn.Bilinear(4, 5, 6),
    nn.Conv1d(4, 6, kernel_size=1),
    nn.Conv2d(4, 6, kernel_size=1),
    nn.Conv3d(4, 6, kernel_size=1),
    nn.LayerNorm(6),
    nn.GroupNorm(1, 6),
    nn.BatchNorm1d(6),
    nn.BatchNorm2d(6),
    nn.BatchNorm3d(6),
)
MODULE_WITHOUT_BIAS = (
    nn.Linear(4, 6, bias=False),
    nn.Bilinear(4, 5, 6, bias=False),
    nn.Conv1d(4, 6, kernel_size=1, bias=False),
    nn.Conv2d(4, 6, kernel_size=1, bias=False),
    nn.Conv3d(4, 6, kernel_size=1, bias=False),
    nn.LayerNorm(6, elementwise_affine=False),
    nn.GroupNorm(1, 6, affine=False),
    nn.BatchNorm1d(6, affine=False),
    nn.BatchNorm2d(6, affine=False),
    nn.BatchNorm3d(6, affine=False),
)


##################################
#     Tests for ConstantBias     #
##################################


def test_constant_bias_str() -> None:
    assert str(ConstantBias()).startswith("ConstantBias(")


@mark.parametrize("value", (1, 2.0))
def test_constant_bias_value(value: Union[int, float]) -> None:
    assert ConstantBias(value=value)._value == value


def test_constant_bias_value_default() -> None:
    assert ConstantBias()._value == 0.0


@mark.parametrize("learnable_only", (True, False))
def test_constant_bias_learnable_only(learnable_only: bool) -> None:
    assert ConstantBias(learnable_only=learnable_only)._learnable_only == learnable_only


def test_constant_bias_learnable_only_default() -> None:
    assert ConstantBias()._learnable_only


@mark.parametrize("log_info", (True, False))
def test_constant_bias_log_info(log_info: bool) -> None:
    assert ConstantBias(log_info=log_info)._log_info == log_info


def test_constant_bias_log_info_default() -> None:
    assert not ConstantBias()._log_info


def test_constant_bias_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    ConstantBias(value=1).initialize(module)
    assert module.weight.equal(torch.zeros(6, 4))
    assert module.bias.equal(torch.ones(6))


def test_constant_bias_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant_init(module, 0.0)
    ConstantBias(value=1).initialize(module)
    assert module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.ones(6))
    assert module[2].weight.data.equal(torch.zeros(6, 6))
    assert module[2].bias.data.equal(torch.ones(6))


@mark.parametrize("value", (1, 2.0))
def test_constant_bias_initialize_value(value: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with patch("gravitorch.nn.init.constant.constant_bias_init") as init:
        ConstantBias(value=value).initialize(module)
        init.assert_called_once_with(
            module=module, value=value, learnable_only=True, log_info=False
        )


@mark.parametrize("learnable_only", (True, False))
def test_constant_bias_initialize_learnable_only(
    learnable_only: bool,
) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.constant.constant_bias_init") as init:
        ConstantBias(learnable_only=learnable_only).initialize(module)
        init.assert_called_once_with(
            module=module, value=0.0, learnable_only=learnable_only, log_info=False
        )


@mark.parametrize("log_info", (True, False))
def test_constant_bias_initialize_log_info(log_info: bool) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.constant.constant_bias_init") as init:
        ConstantBias(log_info=log_info).initialize(module)
        init.assert_called_once_with(
            module=module, value=0.0, learnable_only=True, log_info=log_info
        )


########################################
#     Tests for constant_bias_init     #
########################################


@mark.parametrize("value", (1, 2.0))
def test_constant_bias_init_value(value: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    constant_bias_init(module, value=value)
    assert module.bias.equal(torch.full((6,), value, dtype=torch.float))


@mark.parametrize("module", MODULE_WITH_BIAS)
def test_constant_bias_init_module_with_bias(module: nn.Module) -> None:
    constant_init(module, 1.0)
    assert module.bias.equal(torch.ones(6))
    constant_bias_init(module, value=0.0)
    assert module.bias.equal(torch.zeros(6))


@mark.parametrize("module", MODULE_WITHOUT_BIAS)
def test_constant_bias_init_module_without_bias(module: nn.Module) -> None:
    constant_bias_init(module, value=0.0)
    assert module.bias is None


def test_constant_bias_init_module_multihead_attention() -> None:
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2)
    constant_init(module, 1)
    constant_bias_init(module, value=0.0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_constant_bias_init_module_multihead_attention_add_bias_kv() -> None:
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, add_bias_kv=True)
    constant_init(module, 1)
    constant_bias_init(module, value=0.0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k.equal(torch.zeros(1, 1, 4))
    assert module.bias_v.equal(torch.zeros(1, 1, 4))
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_constant_bias_init_module_multihead_attention_no_bias() -> None:
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False)
    constant_init(module, 1)
    constant_bias_init(module, value=0.0)
    assert module.in_proj_bias is None
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias is None


def test_constant_bias_init_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 1)
    freeze_module(module[1])
    constant_bias_init(module, 0.0, learnable_only=True)
    assert module[0].weight.data.equal(torch.ones(6, 4))  # weights are not initialized
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.ones(6, 6))
    assert module[1].bias.data.equal(torch.ones(6))


def test_constant_bias_init_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant_init(module, 1.0)
    constant_bias_init(module, 0.0)
    assert module[0].weight.data.equal(torch.ones(6, 4))  # weights are not initialized
    assert module[0].bias.data.equal(torch.zeros(6))
    assert module[1].weight.data.equal(torch.ones(6, 6))  # weights are not initialized
    assert module[1].bias.data.equal(torch.zeros(6))


def test_constant_bias_init_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with caplog.at_level(level=logging.INFO):
        constant_bias_init(module, value=0.0, log_info=True)
        assert module.bias.equal(torch.zeros(6))
        assert caplog.messages


def test_constant_bias_init_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with caplog.at_level(level=logging.INFO):
        constant_bias_init(module, value=0.0)
        assert module.bias.equal(torch.zeros(6))
        assert not caplog.messages


###################################
#     Tests for constant_init     #
###################################


def test_constant_init_0() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 0.0)
    assert module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.zeros(6))


def test_constant_init_1() -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    assert module.weight.data.equal(torch.ones(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_constant_init_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    freeze_module(module[1])
    constant_init(module, 0.0, learnable_only=True)
    assert module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert not module[1].bias.data.equal(torch.zeros(6))


def test_constant_init_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    freeze_module(module[1])
    constant_init(module, 0.0, learnable_only=False)
    assert module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_constant_init_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with caplog.at_level(level=logging.INFO):
        constant_init(module, value=0.0, log_info=True)
        assert module.weight.equal(torch.zeros(6, 4))
        assert module.bias.equal(torch.zeros(6))
        assert caplog.messages


def test_constant_init_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant_init(module, 1.0)
    with caplog.at_level(level=logging.INFO):
        constant_init(module, value=0.0)
        assert module.weight.equal(torch.zeros(6, 4))
        assert module.bias.equal(torch.zeros(6))
        assert not caplog.messages
