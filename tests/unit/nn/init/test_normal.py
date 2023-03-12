import logging
import math
from typing import Union
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.nn import freeze_module
from gravitorch.nn.init import TruncNormal, constant, trunc_normal

#################################
#     Tests for TruncNormal     #
#################################


def test_trunc_normal_str() -> None:
    assert str(TruncNormal()).startswith("TruncNormal(")


@mark.parametrize("mean", (1, 2.0))
def test_trunc_normal_init_mean(mean: Union[int, float]) -> None:
    assert TruncNormal(mean=mean)._mean == mean


def test_trunc_normal_init_mean_default() -> None:
    assert TruncNormal()._mean == 0.0


@mark.parametrize("std", (1, 2.0))
def test_trunc_normal_init_std(std: Union[int, float]) -> None:
    assert TruncNormal(std=std)._std == std


def test_trunc_normal_init_std_default() -> None:
    assert TruncNormal()._std == 1.0


@mark.parametrize("min_cutoff", (-1, -2.0))
def test_trunc_normal_init_min_cutoff(min_cutoff: Union[int, float]) -> None:
    assert TruncNormal(min_cutoff=min_cutoff)._min_cutoff == min_cutoff


def test_trunc_normal_init_min_cutoff_default() -> None:
    assert TruncNormal()._min_cutoff == -2.0


@mark.parametrize("max_cutoff", (1, 2.0))
def test_trunc_normal_init_max_cutoff(max_cutoff: Union[int, float]) -> None:
    assert TruncNormal(max_cutoff=max_cutoff)._max_cutoff == max_cutoff


def test_trunc_normal_init_max_cutoff_default() -> None:
    assert TruncNormal()._max_cutoff == 2.0


@mark.parametrize("learnable_only", (True, False))
def test_trunc_normal_init_learnable_only(learnable_only: bool) -> None:
    assert TruncNormal(learnable_only=learnable_only)._learnable_only == learnable_only


def test_trunc_normal_init_learnable_only_default() -> None:
    assert TruncNormal()._learnable_only


@mark.parametrize("log_info", (True, False))
def test_trunc_normal_init_log_info(log_info: bool) -> None:
    assert TruncNormal(log_info=log_info)._log_info == log_info


def test_trunc_normal_init_log_info_default() -> None:
    assert not TruncNormal()._log_info


def test_trunc_normal_initialize_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    TruncNormal().initialize(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert not module.bias.data.equal(torch.zeros(6))


def test_trunc_normal_initialize_sequential() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    constant(module, 0.0)
    TruncNormal().initialize(module)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert not module[0].bias.data.equal(torch.zeros(6))
    assert not module[2].weight.data.equal(torch.zeros(6, 6))
    assert not module[2].bias.data.equal(torch.zeros(6))


@mark.parametrize("mean", (0, 1.0))
def test_trunc_normal_initialize_mean(mean: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(mean=mean).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=mean,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("std", (0.1, 1.0))
def test_trunc_normal_initialize_std(std: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(std=std).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=0.0,
            std=std,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("min_cutoff", (-1, -2.0))
def test_trunc_normal_initialize_min_cutoff(min_cutoff: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(min_cutoff=min_cutoff).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=0.0,
            std=1.0,
            min_cutoff=min_cutoff,
            max_cutoff=2.0,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("max_cutoff", (1, 2.0))
def test_trunc_normal_initialize_max_cutoff(max_cutoff: Union[int, float]) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(max_cutoff=max_cutoff).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=0.0,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=max_cutoff,
            learnable_only=True,
            log_info=False,
        )


@mark.parametrize("learnable_only", (True, False))
def test_trunc_normal_initialize_learnable_only(learnable_only: bool) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(learnable_only=learnable_only).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=0.0,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=learnable_only,
            log_info=False,
        )


@mark.parametrize("log_info", (True, False))
def test_trunc_normal_initialize_log_info(log_info: bool) -> None:
    module = nn.Linear(4, 6)
    with patch("gravitorch.nn.init.normal.trunc_normal") as trunc_normal:
        TruncNormal(log_info=log_info).initialize(module)
        trunc_normal.assert_called_once_with(
            module=module,
            mean=0.0,
            std=1.0,
            min_cutoff=-2.0,
            max_cutoff=2.0,
            learnable_only=True,
            log_info=log_info,
        )


##################################
#     Tests for trunc_normal     #
##################################


def test_trunc_normal_linear() -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    trunc_normal(module)
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert not module.bias.data.equal(torch.zeros(6))


@mark.parametrize("mean", (0.0, 0.5, 1.0))
def test_trunc_normal_mean(mean: float) -> None:
    module = nn.Linear(200, 200)
    constant(module, 0.0)
    trunc_normal(module, mean=mean, min_cutoff=mean - 2, max_cutoff=mean + 2)
    assert math.isclose(module.weight.data.mean().item(), mean, abs_tol=0.05)


@mark.parametrize("std", (0.1, 0.5, 1.0))
def test_trunc_normal_std(std: float) -> None:
    module = nn.Linear(200, 200)
    constant(module, 0.0)
    trunc_normal(module, std=std, min_cutoff=-10, max_cutoff=10)
    assert math.isclose(module.weight.data.std().item(), std, rel_tol=0.05)


@mark.parametrize("min_cutoff", (-1.0, -0.5, -0.1))
def test_trunc_normal_min_cutoff(min_cutoff: float) -> None:
    module = nn.Linear(200, 200)
    constant(module, 0.0)
    trunc_normal(module, min_cutoff=min_cutoff)
    assert module.weight.data.min() >= min_cutoff


@mark.parametrize("max_cutoff", (1.0, 0.5, 0.1))
def test_trunc_normal_max_cutoff(max_cutoff: float) -> None:
    module = nn.Linear(200, 200)
    constant(module, 0.0)
    trunc_normal(module, max_cutoff=max_cutoff)
    assert module.weight.data.max() <= max_cutoff


def test_trunc_normal_learnable_only_true() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    trunc_normal(module, learnable_only=True)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert not module[0].bias.data.equal(torch.zeros(6))
    # The second linear should not be initialized because it is frozen
    assert module[1].weight.data.equal(torch.zeros(6, 6))
    assert module[1].bias.data.equal(torch.zeros(6))


def test_trunc_normal_learnable_only_false() -> None:
    module = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    constant(module, 0.0)
    freeze_module(module[1])
    trunc_normal(module, learnable_only=False)
    assert not module[0].weight.data.equal(torch.zeros(6, 4))
    assert not module[0].bias.data.equal(torch.zeros(6))
    assert not module[1].weight.data.equal(torch.zeros(6, 6))
    assert not module[1].bias.data.equal(torch.zeros(6))


def test_trunc_normal_log_info_true(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        trunc_normal(module, log_info=True)
        assert caplog.messages
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert not module.bias.data.equal(torch.zeros(6))


def test_trunc_normal_log_info_false(caplog: LogCaptureFixture) -> None:
    module = nn.Linear(4, 6)
    constant(module, 0.0)
    with caplog.at_level(level=logging.INFO):
        trunc_normal(module)
        assert not caplog.messages
    assert not module.weight.data.equal(torch.zeros(6, 4))
    assert not module.bias.data.equal(torch.zeros(6))
