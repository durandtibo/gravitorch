import logging

import torch
from pytest import LogCaptureFixture, mark
from torch import nn

from gravitorch.nn import compute_parameter_stats, show_parameter_stats
from gravitorch.utils import get_available_devices

#############################################
#     Tests for compute_parameter_stats     #
#############################################


def test_compute_parameter_stats_linear() -> None:
    stats = compute_parameter_stats(nn.Linear(4, 6))
    assert len(stats) == 3
    assert stats[0] == ["parameter", "mean", "median", "std", "min", "max", "learnable"]
    assert stats[1][0] == "weight"
    assert isinstance(stats[1][1], float)  # mean
    assert isinstance(stats[1][2], float)  # median
    assert isinstance(stats[1][3], float)  # std
    assert isinstance(stats[1][4], float)  # min
    assert isinstance(stats[1][5], float)  # max
    assert isinstance(stats[1][6], bool)  # learnable


def test_compute_parameter_stats_empty_tensor() -> None:
    assert compute_parameter_stats(nn.Linear(4, 0)) == [
        ["parameter", "mean", "median", "std", "min", "max", "learnable"]
    ]


##########################################
#     Tests for show_parameter_stats     #
##########################################


@mark.parametrize("device", get_available_devices())
def test_show_parameter_stats_linear(device: str, caplog: LogCaptureFixture) -> None:
    device = torch.device(device)
    with caplog.at_level(logging.DEBUG):
        show_parameter_stats(nn.Linear(4, 6).to(device=device))
        assert len(caplog.messages) > 0
