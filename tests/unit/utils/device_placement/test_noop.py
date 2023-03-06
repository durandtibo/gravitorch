import torch

from gravitorch.utils.device_placement import NoOpDevicePlacement

#########################################
#     Tests for NoOpDevicePlacement     #
#########################################


def test_noop_device_placement_str() -> None:
    assert str(NoOpDevicePlacement()) == "NoOpDevicePlacement()"


def test_noop_device_placement_send() -> None:
    device_placement = NoOpDevicePlacement()
    x = torch.ones(2, 3)
    assert device_placement.send(x) is x
