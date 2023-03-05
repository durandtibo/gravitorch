import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark
from torch import nn
from torch.optim import SGD, Adagrad, Adam, Rprop

from gravitorch.optimizers.utils import (
    get_learning_rate_per_group,
    get_weight_decay_per_group,
    log_optimizer_parameters_per_group,
    show_optimizer_parameters_per_group,
)
from gravitorch.utils.exp_trackers import EpochStep

################################################
#     Tests of get_learning_rate_per_group     #
################################################


@mark.parametrize("lr", (0.01, 0.0001))
def test_get_learning_rate_per_group_single_value(lr: float):
    model = nn.Linear(4, 6)
    optimizer = SGD(model.parameters(), lr=lr)
    assert get_learning_rate_per_group(optimizer) == {0: lr}


def test_get_learning_rate_per_group_multiple_values() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    optimizer = SGD(
        [{"params": model[0].parameters(), "lr": 0.001}, {"params": model[1].parameters()}], lr=0.01
    )
    assert get_learning_rate_per_group(optimizer) == {0: 0.001, 1: 0.01}


###############################################
#     Tests of get_weight_decay_per_group     #
###############################################


@mark.parametrize("weight_decay", (0.0001, 0.00001))
def test_get_weight_decays_single_value(weight_decay: float):
    model = nn.Linear(4, 6)
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)
    assert get_weight_decay_per_group(optimizer) == {0: weight_decay}


def test_get_weight_decays_multiple_value() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    optimizer = SGD(
        [
            {"params": model[0].parameters(), "weight_decay": 0.0002},
            {"params": model[1].parameters()},
        ],
        lr=0.01,
        weight_decay=0.0001,
    )
    assert get_weight_decay_per_group(optimizer) == {0: 0.0002, 1: 0.0001}


def test_get_weight_decays_no_weight_decay() -> None:
    model = nn.Linear(4, 6)
    optimizer = Rprop(model.parameters(), lr=0.01)
    assert get_weight_decay_per_group(optimizer) == {}


########################################################
#     Tests of show_optimizer_parameters_per_group     #
########################################################


def test_show_optimizer_parameters_per_group_sgd_1_group(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        model = nn.Linear(4, 6)
        optimizer = SGD(model.parameters(), lr=0.01)
        show_optimizer_parameters_per_group(optimizer)
        assert caplog.records[0].message == (
            "Optimizer: parameters per group\n"
            "╒═════════╤═════════════╤══════════════════╤═══════════╤══════╤════════════╤════════════╤════════════╤════════════════╕\n"  # noqa: E501,B950
            "│         │   dampening │ differentiable   │ foreach   │   lr │ maximize   │   momentum │ nesterov   │   weight_decay │\n"  # noqa: E501,B950
            "╞═════════╪═════════════╪══════════════════╪═══════════╪══════╪════════════╪════════════╪════════════╪════════════════╡\n"  # noqa: E501,B950
            "│ Group 0 │           0 │ False            │           │ 0.01 │ False      │          0 │ False      │              0 │\n"  # noqa: E501,B950
            "╘═════════╧═════════════╧══════════════════╧═══════════╧══════╧════════════╧════════════╧════════════╧════════════════╛"  # noqa: E501,B950
        )


def test_show_optimizer_parameters_per_group_sgd_2_groups(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
        optimizer = SGD(
            [
                {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.0002},
                {"params": model[1].parameters()},
            ],
            lr=0.01,
            weight_decay=0.0001,
        )
        show_optimizer_parameters_per_group(optimizer)
        assert caplog.records[0].message == (
            "Optimizer: parameters per group\n"
            "╒═════════╤═════════════╤══════════════════╤═══════════╤═══════╤════════════╤════════════╤════════════╤════════════════╕\n"  # noqa: E501,B950
            "│         │   dampening │ differentiable   │ foreach   │    lr │ maximize   │   momentum │ nesterov   │   weight_decay │\n"  # noqa: E501,B950
            "╞═════════╪═════════════╪══════════════════╪═══════════╪═══════╪════════════╪════════════╪════════════╪════════════════╡\n"  # noqa: E501,B950
            "│ Group 0 │           0 │ False            │           │ 0.001 │ False      │          0 │ False      │         0.0002 │\n"  # noqa: E501,B950
            "├─────────┼─────────────┼──────────────────┼───────────┼───────┼────────────┼────────────┼────────────┼────────────────┤\n"  # noqa: E501,B950
            "│ Group 1 │           0 │ False            │           │ 0.01  │ False      │          0 │ False      │         0.0001 │\n"  # noqa: E501,B950
            "╘═════════╧═════════════╧══════════════════╧═══════════╧═══════╧════════════╧════════════╧════════════╧════════════════╛"  # noqa: E501,B950
        )


def test_show_optimizer_parameters_per_group_adam_1_group(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        model = nn.Linear(4, 6)
        optimizer = Adam(model.parameters(), lr=0.01)
        show_optimizer_parameters_per_group(optimizer)
        assert caplog.records[0].message == (
            "Optimizer: parameters per group\n"
            "╒═════════╤═══════════╤══════════════╤══════════════╤══════════════════╤═══════╤═══════════╤═════════╤══════╤════════════╤════════════════╕\n"  # noqa: E501,B950
            "│         │ amsgrad   │ betas        │ capturable   │ differentiable   │   eps │ foreach   │ fused   │   lr │ maximize   │   weight_decay │\n"  # noqa: E501,B950
            "╞═════════╪═══════════╪══════════════╪══════════════╪══════════════════╪═══════╪═══════════╪═════════╪══════╪════════════╪════════════════╡\n"  # noqa: E501,B950
            "│ Group 0 │ False     │ (0.9, 0.999) │ False        │ False            │ 1e-08 │           │ False   │ 0.01 │ False      │              0 │\n"  # noqa: E501,B950
            "╘═════════╧═══════════╧══════════════╧══════════════╧══════════════════╧═══════╧═══════════╧═════════╧══════╧════════════╧════════════════╛"  # noqa: E501,B950
        )


def test_show_optimizer_parameters_per_group_adagrad_1_group(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        model = nn.Linear(4, 6)
        optimizer = Adagrad(model.parameters(), lr=0.01)
        show_optimizer_parameters_per_group(optimizer)
        assert caplog.records[0].message == (
            "Optimizer: parameters per group\n"
            "╒═════════╤═══════╤═══════════╤═════════════════════════════╤══════╤════════════╤════════════╤════════════════╕\n"  # noqa: E501,B950
            "│         │   eps │ foreach   │   initial_accumulator_value │   lr │   lr_decay │ maximize   │   weight_decay │\n"  # noqa: E501,B950
            "╞═════════╪═══════╪═══════════╪═════════════════════════════╪══════╪════════════╪════════════╪════════════════╡\n"  # noqa: E501,B950
            "│ Group 0 │ 1e-10 │           │                           0 │ 0.01 │          0 │ False      │              0 │\n"  # noqa: E501,B950
            "╘═════════╧═══════╧═══════════╧═════════════════════════════╧══════╧════════════╧════════════╧════════════════╛"  # noqa: E501,B950
        )


#######################################################
#     Tests of log_optimizer_parameters_per_group     #
#######################################################


def test_log_optimizer_parameters_per_group_sgd_1_group() -> None:
    model = nn.Linear(4, 6)
    optimizer = SGD(model.parameters(), lr=0.01)
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine)
    engine.log_metrics.assert_called_once_with(
        {
            "optimizer.group0.dampening": 0,
            "optimizer.group0.differentiable": False,
            "optimizer.group0.lr": 0.01,
            "optimizer.group0.maximize": False,
            "optimizer.group0.momentum": 0,
            "optimizer.group0.nesterov": False,
            "optimizer.group0.weight_decay": 0,
        },
        step=None,
    )


def test_log_optimizer_parameters_per_group_sgd_1_group_with_step() -> None:
    model = nn.Linear(4, 6)
    optimizer = SGD(model.parameters(), lr=0.01)
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine, step=EpochStep(2))
    engine.log_metrics.assert_called_once_with(
        {
            "optimizer.group0.dampening": 0,
            "optimizer.group0.differentiable": False,
            "optimizer.group0.lr": 0.01,
            "optimizer.group0.maximize": False,
            "optimizer.group0.momentum": 0,
            "optimizer.group0.nesterov": False,
            "optimizer.group0.weight_decay": 0,
        },
        step=EpochStep(2),
    )


def test_log_optimizer_parameters_per_group_sgd_1_group_with_prefix() -> None:
    model = nn.Linear(4, 6)
    optimizer = SGD(model.parameters(), lr=0.01)
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine, prefix="train/")
    engine.log_metrics.assert_called_once_with(
        {
            "train/optimizer.group0.dampening": 0,
            "train/optimizer.group0.differentiable": False,
            "train/optimizer.group0.lr": 0.01,
            "train/optimizer.group0.maximize": False,
            "train/optimizer.group0.momentum": 0,
            "train/optimizer.group0.nesterov": False,
            "train/optimizer.group0.weight_decay": 0,
        },
        step=None,
    )


def test_log_optimizer_parameters_per_group_sgd_2_groups() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6))
    optimizer = SGD(
        [
            {"params": model[0].parameters(), "lr": 0.001, "weight_decay": 0.0002},
            {"params": model[1].parameters()},
        ],
        lr=0.01,
        weight_decay=0.0001,
    )
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine)
    engine.log_metrics.assert_called_once_with(
        {
            "optimizer.group0.dampening": 0,
            "optimizer.group0.differentiable": False,
            "optimizer.group0.lr": 0.001,
            "optimizer.group0.maximize": False,
            "optimizer.group0.momentum": 0,
            "optimizer.group0.nesterov": False,
            "optimizer.group0.weight_decay": 0.0002,
            "optimizer.group1.dampening": 0,
            "optimizer.group1.differentiable": False,
            "optimizer.group1.lr": 0.01,
            "optimizer.group1.maximize": False,
            "optimizer.group1.momentum": 0,
            "optimizer.group1.nesterov": False,
            "optimizer.group1.weight_decay": 0.0001,
        },
        step=None,
    )


def test_log_optimizer_parameters_per_group_adam_1_group() -> None:
    model = nn.Linear(4, 6)
    optimizer = Adam(model.parameters(), lr=0.01)
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine)
    engine.log_metrics.assert_called_once_with(
        {
            "optimizer.group0.amsgrad": False,
            "optimizer.group0.betas.0": 0.9,
            "optimizer.group0.betas.1": 0.999,
            "optimizer.group0.capturable": False,
            "optimizer.group0.differentiable": False,
            "optimizer.group0.eps": 1e-08,
            "optimizer.group0.fused": False,
            "optimizer.group0.lr": 0.01,
            "optimizer.group0.maximize": False,
            "optimizer.group0.weight_decay": 0,
        },
        step=None,
    )


def test_log_optimizer_parameters_per_group_adagrad_1_group() -> None:
    model = nn.Linear(4, 6)
    optimizer = Adagrad(model.parameters(), lr=0.01)
    engine = Mock()
    log_optimizer_parameters_per_group(optimizer, engine)
    engine.log_metrics.assert_called_once_with(
        {
            "optimizer.group0.eps": 1e-10,
            "optimizer.group0.initial_accumulator_value": 0,
            "optimizer.group0.lr": 0.01,
            "optimizer.group0.lr_decay": 0,
            "optimizer.group0.maximize": False,
            "optimizer.group0.weight_decay": 0,
        },
        step=None,
    )
