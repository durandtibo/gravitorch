from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.models.utils import analyze_module_architecture
from gravitorch.models.utils.architecture import (
    analyze_model_architecture,
    analyze_network_architecture,
)
from gravitorch.utils.exp_trackers import EpochStep

#################################################
#     Tests for analyze_module_architecture     #
#################################################


def test_analyze_module_architecture_not_a_module(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        analyze_module_architecture("abc")
        assert len(caplog.messages) == 0


def test_analyze_module_architecture_without_engine(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        analyze_module_architecture(nn.Linear(4, 6))
        assert len(caplog.messages) == 3


def test_analyze_module_architecture_with_engine() -> None:
    engine = Mock(epoch=0)
    analyze_module_architecture(nn.Linear(4, 6), engine)
    engine.log_metrics.assert_called_once_with(
        {
            "num_parameters": 30,
            "num_learnable_parameters": 30,
        },
        step=EpochStep(0),
    )


@mark.parametrize("prefix", ("name.", "module.network."))
@patch("gravitorch.models.utils.architecture.num_parameters", lambda *args, **kwargs: 12)
@patch(
    "gravitorch.models.utils.architecture.num_learnable_parameters",
    lambda *args, **kwargs: 10,
)
def test_analyze_module_architecture_with_engine_prefix(prefix: str) -> None:
    engine = Mock(epoch=0)
    analyze_module_architecture(nn.Linear(4, 6), engine, prefix)
    engine.log_metrics.assert_called_once_with(
        {
            f"{prefix}num_parameters": 12,
            f"{prefix}num_learnable_parameters": 10,
        },
        step=EpochStep(0),
    )


################################################
#     Tests for analyze_model_architecture     #
################################################


def test_analyze_model_architecture() -> None:
    engine = Mock(epoch=0)
    analyze_model_architecture(nn.Linear(4, 6), engine)
    engine.log_metrics.assert_called_once_with(
        {
            "model.num_parameters": 30,
            "model.num_learnable_parameters": 30,
        },
        step=EpochStep(0),
    )


##################################################
#     Tests for analyze_network_architecture     #
##################################################


def test_analyze_network_architecture() -> None:
    engine = Mock(epoch=0)
    model = Mock(network=nn.Linear(4, 6))
    analyze_network_architecture(model, engine)
    engine.log_metrics.assert_called_once_with(
        {
            "model.network.num_parameters": 30,
            "model.network.num_learnable_parameters": 30,
        },
        step=EpochStep(0),
    )


def test_analyze_network_architecture_no_network() -> None:
    engine = Mock(epoch=0)
    analyze_network_architecture(nn.Linear(4, 6), engine)
    engine.log_metrics.assert_not_called()
