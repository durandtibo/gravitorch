import math
from typing import Union
from unittest.mock import Mock, patch

import torch
from pytest import mark
from torch import nn

from gravitorch.engines import BaseEngine
from gravitorch.nn import freeze_module
from gravitorch.utils.parameter_initializers import (
    TruncNormalParameterInitializer,
    recursive_constant_,
    recursive_trunc_normal_,
)

#####################################################
#     Tests for TruncNormalParameterInitializer     #
#####################################################


def test_trunc_normal_parameter_initializer_str():
    assert str(TruncNormalParameterInitializer()).startswith("TruncNormalParameterInitializer(")


@mark.parametrize("mean", (1, 2.0))
def test_trunc_normal_parameter_initializer_mean(mean: Union[int, float]):
    assert TruncNormalParameterInitializer(mean=mean)._mean == mean


def test_trunc_normal_parameter_initializer_mean_default():
    assert TruncNormalParameterInitializer()._mean == 0.0


@mark.parametrize("std", (1, 2.0))
def test_trunc_normal_parameter_initializer_std(std: Union[int, float]):
    assert TruncNormalParameterInitializer(std=std)._std == std


def test_trunc_normal_parameter_initializer_std_default():
    assert TruncNormalParameterInitializer()._std == 1.0


@mark.parametrize("min_cutoff", (-1, -2.0))
def test_trunc_normal_parameter_initializer_min_cutoff(min_cutoff: Union[int, float]):
    assert TruncNormalParameterInitializer(min_cutoff=min_cutoff)._min_cutoff == min_cutoff


def test_trunc_normal_parameter_initializer_min_cutoff_default():
    assert TruncNormalParameterInitializer()._min_cutoff == -2.0


@mark.parametrize("max_cutoff", (1, 2.0))
def test_trunc_normal_parameter_initializer_max_cutoff(max_cutoff: Union[int, float]):
    assert TruncNormalParameterInitializer(max_cutoff=max_cutoff)._max_cutoff == max_cutoff


def test_trunc_normal_parameter_initializer_max_cutoff_default():
    assert TruncNormalParameterInitializer()._max_cutoff == 2.0


@mark.parametrize("learnable_only", (True, False))
def test_trunc_normal_parameter_initializer_learnable_only(learnable_only: bool):
    assert (
        TruncNormalParameterInitializer(learnable_only=learnable_only)._learnable_only
        == learnable_only
    )


def test_trunc_normal_parameter_initializer_learnable_only_default():
    assert TruncNormalParameterInitializer()._learnable_only


@mark.parametrize("show_stats", (True, False))
def test_trunc_normal_parameter_initializer_show_stats(show_stats: bool):
    assert TruncNormalParameterInitializer(show_stats=show_stats)._show_stats == show_stats


def test_trunc_normal_parameter_initializer_show_stats_default():
    assert TruncNormalParameterInitializer()._show_stats


def test_trunc_normal_parameter_initializer_initialize_linear():
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 0)
    TruncNormalParameterInitializer().initialize(engine)
    assert not engine.model.weight.data.equal(torch.zeros(6, 4))
    assert not engine.model.bias.data.equal(torch.zeros(6))


def test_trunc_normal_parameter_initializer_initialize_sequential():
    engine = Mock(spec=BaseEngine, model=nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)))
    recursive_constant_(engine.model, 0)
    TruncNormalParameterInitializer().initialize(engine)
    assert not engine.model[0].weight.data.equal(torch.zeros(6, 4))
    assert not engine.model[0].bias.data.equal(torch.zeros(6))
    assert not engine.model[2].weight.data.equal(torch.zeros(6, 6))
    assert not engine.model[2].bias.data.equal(torch.zeros(6))


@mark.parametrize("mean", (0, 1.0))
def test_trunc_normal_parameter_initializer_initialize_mean(mean: Union[int, float]):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch(
        "gravitorch.utils.parameter_initializers.truncated_normal.recursive_trunc_normal_"
    ) as trunc_normal:
        TruncNormalParameterInitializer(mean=mean).initialize(engine)
        trunc_normal.assert_called_once_with(
            module=engine.model,
            mean=mean,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=True,
        )


@mark.parametrize("std", (0.1, 1.0))
def test_trunc_normal_parameter_initializer_initialize_std(std: Union[int, float]):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch(
        "gravitorch.utils.parameter_initializers.truncated_normal.recursive_trunc_normal_"
    ) as trunc_normal:
        TruncNormalParameterInitializer(std=std).initialize(engine)
        trunc_normal.assert_called_once_with(
            module=engine.model,
            mean=0.0,
            std=std,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=True,
        )


@mark.parametrize("min_cutoff", (-1, -2.0))
def test_trunc_normal_parameter_initializer_initialize_min_cutoff(min_cutoff: Union[int, float]):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch(
        "gravitorch.utils.parameter_initializers.truncated_normal.recursive_trunc_normal_"
    ) as trunc_normal:
        TruncNormalParameterInitializer(min_cutoff=min_cutoff).initialize(engine)
        trunc_normal.assert_called_once_with(
            module=engine.model,
            mean=0.0,
            std=1.0,
            min_cutoff=min_cutoff,
            max_cutoff=2.0,
            learnable_only=True,
        )


@mark.parametrize("max_cutoff", (1, 2.0))
def test_trunc_normal_parameter_initializer_initialize_max_cutoff(max_cutoff: Union[int, float]):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    with patch(
        "gravitorch.utils.parameter_initializers.truncated_normal.recursive_trunc_normal_"
    ) as trunc_normal:
        TruncNormalParameterInitializer(max_cutoff=max_cutoff).initialize(engine)
        trunc_normal.assert_called_once_with(
            module=engine.model,
            mean=0.0,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=max_cutoff,
            learnable_only=True,
        )


@mark.parametrize("learnable_only", (True, False))
def test_trunc_normal_parameter_initializer_initialize_learnable_only(learnable_only: bool):
    engine = Mock(spec=BaseEngine, model=nn.Linear(4, 6))
    recursive_constant_(engine.model, 1)
    with patch(
        "gravitorch.utils.parameter_initializers.truncated_normal.recursive_trunc_normal_"
    ) as trunc_normal:
        TruncNormalParameterInitializer(learnable_only=learnable_only).initialize(engine)
        trunc_normal.assert_called_once_with(
            module=engine.model,
            mean=0.0,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=learnable_only,
        )


#############################################
#     Tests for recursive_trunc_normal_     #
#############################################


def test_recursive_trunc_normal_linear():
    module = nn.Linear(4, 6)
    recursive_constant_(module, 0)
    recursive_trunc_normal_(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert not module.bias.data.equal(torch.zeros(6))


@mark.parametrize("mean", (0.0, 0.5, 1.0))
def test_recursive_trunc_normal_mean(mean: float):
    module = nn.Linear(200, 200)
    recursive_constant_(module, 0)
    recursive_trunc_normal_(module, mean=mean, min_cutoff=mean - 2, max_cutoff=mean + 2)
    assert math.isclose(module.weight.data.mean().item(), mean, abs_tol=0.05)


@mark.parametrize("std", (0.1, 0.5, 1.0))
def test_recursive_trunc_normal_std(std: float):
    module = nn.Linear(200, 200)
    recursive_constant_(module, 0)
    recursive_trunc_normal_(module, std=std, min_cutoff=-10, max_cutoff=10)
    assert math.isclose(module.weight.data.std().item(), std, rel_tol=0.05)


@mark.parametrize("min_cutoff", (-1.0, -0.5, -0.1))
def test_recursive_trunc_normal_min_cutoff(min_cutoff: float):
    module = nn.Linear(200, 200)
    recursive_constant_(module, 0)
    recursive_trunc_normal_(module, min_cutoff=min_cutoff)
    assert module.weight.data.min() >= min_cutoff


@mark.parametrize("max_cutoff", (1.0, 0.5, 0.1))
def test_recursive_trunc_normal_max_cutoff(max_cutoff: float):
    module = nn.Linear(200, 200)
    recursive_constant_(module, 0)
    recursive_trunc_normal_(module, max_cutoff=max_cutoff)
    assert module.weight.data.max() <= max_cutoff


def test_recursive_trunc_normal_sequential_learnable_only_true():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_trunc_normal_(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert not module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_recursive_trunc_normal_sequential_learnable_only_false():
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    recursive_constant_(module, 0)
    freeze_module(module[1])
    recursive_trunc_normal_(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert not module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert not module[1].bias.data.equal(torch.zeros(6))
