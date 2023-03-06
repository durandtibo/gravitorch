from unittest.mock import patch

import torch

from gravitorch.utils.device_placement import AutoDevicePlacement

#########################################
#     Tests for AutoDevicePlacement     #
#########################################


@patch(
    "gravitorch.utils.device_placement.auto.dist.device",
    lambda *args, **kwargs: torch.device("cpu"),
)
def test_auto_device_placement_device_cpu() -> None:
    assert AutoDevicePlacement().device == torch.device("cpu")


@patch(
    "gravitorch.utils.device_placement.auto.dist.device",
    lambda *args, **kwargs: torch.device("cuda:0"),
)
def test_auto_device_placement_device_cuda() -> None:
    assert AutoDevicePlacement().device == torch.device("cuda:0")
