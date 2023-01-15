import copy
import logging
from typing import Union
from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.engines import BaseEngine
from gravitorch.nn import freeze_module
from gravitorch.utils.parameter_initializers import (
    ConstantBiasParameterInitializer,
    bias_constant_,
    recursive_bias_constant_,
    recursive_constant_,
)

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


######################################################
#     Tests for ConstantBiasParameterInitializer     #
######################################################


def test_constant_bias_parameter_initializer_str():
    assert str(ConstantBiasParameterInitializer()).startswith("ConstantBiasParameterInitializer(")


@mark.parametrize("value", (1, 2.0))
def test_constant_bias_parameter_initializer_value(value: Union[int, float]):
    assert ConstantBiasParameterInitializer(value=value)._value == value


def test_constant_bias_parameter_initializer_value_default():
    assert ConstantBiasParameterInitializer()._value == 0.0


@mark.parametrize("learnable_only", (True, False))
def test_constant_bias_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        ConstantBiasParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_constant_bias_parameter_initializer_learnable_only_default():
    assert ConstantBiasParameterInitializer()._learnable_only


@mark.parametrize("log_info", (True, False))
def test_constant_bias_parameter_initializer_log_info(log_info: bool):
    assert ConstantBiasParameterInitializer(log_info=log_info)._log_info == log_info


def test_constant_bias_parameter_initializer_log_info_default():
    assert not ConstantBiasParameterInitializer()._log_info


@mark.parametrize("show_stats", (True, False))
def test_constant_bias_parameter_initializer_show_stats(show_stats: bool):
    assert ConstantBiasParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_constant_bias_parameter_initializer_show_stats_default():
    assert ConstantBiasParameterInitializer()._show_stats


def test_constant_bias_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 0)
    ConstantBiasParameterInitializer(value=1).initialize(engine)
    assert engine.model.weight.equal(torch.zeros(6, 4))
    assert engine.model.bias.equal(torch.ones(6))


def test_constant_bias_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 0)
    ConstantBiasParameterInitializer(value=1).initialize(engine)
    assert engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert engine.model[0].bias.data.equal(torch.ones(6))
    assert engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert engine.model[2].bias.data.equal(torch.ones(6))


@mark.parametrize("value", (1, 2.0))
def test_constant_bias_parameter_initializer_initialize_value(value: Union[int, float]):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch("gravitorch.utils.parameter_initializers.constant.recursive_bias_constant_") as init:
        ConstantBiasParameterInitializer(value=value).initialize(engine)
        init.assert_called_once_with(
            module=engine.model, value=value, learnable_only=True, log_info=False
        )


@mark.parametrize("learnable_only", (True, False))
def test_constant_bias_parameter_initializer_initialize_learnable_only(learnable_only: bool):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch("gravitorch.utils.parameter_initializers.constant.recursive_bias_constant_") as init:
        ConstantBiasParameterInitializer(learnable_only=learnable_only).initialize(engine)
        init.assert_called_once_with(
            module=engine.model, value=0.0, learnable_only=learnable_only, log_info=False
        )


@mark.parametrize("log_info", (True, False))
def test_constant_bias_parameter_initializer_initialize_log_info(log_info: bool):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch("gravitorch.utils.parameter_initializers.constant.recursive_bias_constant_") as init:
        ConstantBiasParameterInitializer(log_info=log_info).initialize(engine)
        init.assert_called_once_with(
            module=engine.model, value=0.0, learnable_only=True, log_info=log_info
        )


####################################
#     Tests for bias_constant_     #
####################################


@mark.parametrize("value", (1, 2.0))
def test_bias_constant_value(value: Union[int, float]):
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    bias_constant_(module, value=value)
    assert module.bias.equal(torch.full((6,), value, dtype=torch.float))


@mark.parametrize("module", MODULE_WITH_BIAS)
def test_bias_constant_module_with_bias(module: nn.Module):
    recursive_constant_(module, 1)
    assert module.bias.equal(torch.ones(6))
    bias_constant_(module, value=0)
    assert module.bias.equal(torch.zeros(6))


@mark.parametrize("module", MODULE_WITHOUT_BIAS)
def test_bias_constant_module_without_bias(module: nn.Module):
    bias_constant_(module, value=0)
    assert module.bias is None


def test_bias_constant_module_multihead_attention():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2)
    recursive_constant_(module, 1)
    bias_constant_(module, value=0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_bias_constant_module_multihead_attention_add_bias_kv():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, add_bias_kv=True)
    recursive_constant_(module, 1)
    bias_constant_(module, value=0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k.equal(torch.zeros(1, 1, 4))
    assert module.bias_v.equal(torch.zeros(1, 1, 4))
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_bias_constant_module_multihead_attention_no_bias():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False)
    recursive_constant_(module, 1)
    bias_constant_(module, value=0)
    assert module.in_proj_bias is None
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias is None


def test_bias_constant_module_sequential():
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6), nn.Linear(6, 1))
    recursive_constant_(module, 1)
    bias_constant_(module, value=0)
    # Not a valid module so it should not initialize the biases
    assert not module[0].bias.equal(torch.zeros(6))
    assert not module[2].bias.equal(torch.zeros(6))
    assert not module[3].bias.equal(torch.zeros(1))


##############################################
#     Tests for recursive_bias_constant_     #
##############################################


@mark.parametrize("value", (1, 2.0))
def test_recursive_bias_constant_value(value: Union[int, float]):
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_bias_constant_(module, value=value)
    assert module.bias.equal(torch.full((6,), value, dtype=torch.float))


@mark.parametrize("module", MODULE_WITH_BIAS)
def test_recursive_bias_constant_module_with_bias(module: nn.Module):
    recursive_constant_(module, 1)
    assert module.bias.equal(torch.ones(6))
    recursive_bias_constant_(module, value=0)
    assert module.bias.equal(torch.zeros(6))


@mark.parametrize("module", MODULE_WITHOUT_BIAS)
def test_recursive_bias_constant_module_without_bias(module: nn.Module):
    recursive_bias_constant_(module, value=0)
    assert module.bias is None


def test_recursive_bias_constant_module_multihead_attention():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2)
    recursive_constant_(module, 1)
    recursive_bias_constant_(module, value=0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_recursive_bias_constant_module_multihead_attention_add_bias_kv():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, add_bias_kv=True)
    recursive_constant_(module, 1)
    recursive_bias_constant_(module, value=0)
    assert module.in_proj_bias.equal(torch.zeros(12))
    assert module.bias_k.equal(torch.zeros(1, 1, 4))
    assert module.bias_v.equal(torch.zeros(1, 1, 4))
    assert module.out_proj.bias.equal(torch.zeros(4))


def test_recursive_bias_constant_module_multihead_attention_no_bias():
    module = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False)
    recursive_constant_(module, 1)
    recursive_bias_constant_(module, value=0)
    assert module.in_proj_bias is None
    assert module.bias_k is None
    assert module.bias_v is None
    assert module.out_proj.bias is None


def test_recursive_bias_constant_0_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 1)
    freeze_module(module[1])
    recursive_bias_constant_(module, 0, learnable_only=True)
    assert module[0].weight.data.equal(torch.ones(6, 4))  # weights are not initialized
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.ones(6, 6))
    assert module[1].bias.data.equal(torch.ones(6))


def test_recursive_bias_constant_0_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 1)
    recursive_bias_constant_(module, 0)
    assert module[0].weight.data.equal(torch.ones(6, 4))  # weights are not initialized
    assert module[0].bias.data.equal(torch.zeros(6))
    assert module[1].weight.data.equal(torch.ones(6, 6))  # weights are not initialized
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_bias_constant_log_info_true(caplog: LogCaptureFixture):
    module = nn.Linear(4, 6)
    recursive_constant_(module, 1)
    with caplog.at_level(level=logging.INFO):
        recursive_bias_constant_(module, value=0, log_info=True)
        assert module.bias.equal(torch.zeros(6))
        assert caplog.messages


def test_recursive_bias_constant_log_info_false(caplog: LogCaptureFixture):
    module = nn.Linear(4, 6)
    recursive_constant_(module, 1)
    with caplog.at_level(level=logging.INFO):
        recursive_bias_constant_(module, value=0)
        assert module.bias.equal(torch.zeros(6))
        assert not caplog.messages


###########################################################################
#     Consistency between bias_constant_ and recursive_bias_constant_     #
###########################################################################


@mark.parametrize("module", MODULE_WITH_BIAS + MODULE_WITHOUT_BIAS)
def test_consistency_bias_constant_and_recursive_bias_constant(module: nn.Module):
    # Verify bias_constant_ and recursive_bias_constant_ have consistent results.
    recursive_constant_(module, 1)
    module1 = copy.deepcopy(module)
    module2 = copy.deepcopy(module)
    bias_constant_(module1, value=2)
    recursive_bias_constant_(module2, value=2)
    assert objects_are_equal(module1.state_dict(), module2.state_dict())


#########################################
#     Tests for recursive_constant_     #
#########################################


def test_recursive_constant_0():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    assert module.weight.data.equal(torch.zeros(6, 4))
    assert module.bias.data.equal(torch.zeros(6))


def test_recursive_constant_1():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 1)
    assert module.weight.data.equal(torch.ones(6, 4))
    assert module.bias.data.equal(torch.ones(6))


def test_recursive_constant_0_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    freeze_module(module[1])
    recursive_constant_(module, 0, learnable_only=True)
    assert module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert not module[1].bias.data.equal(torch.zeros(6))


def test_recursive_constant_0_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    freeze_module(module[1])
    recursive_constant_(module, 0, learnable_only=False)
    assert module[0].weight.data.equal(torch.zeros(6, 4))
    assert module[0].bias.data.equal(torch.zeros(6))
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))
