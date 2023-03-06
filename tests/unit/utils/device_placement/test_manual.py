import torch
from pytest import mark

from gravitorch.utils import get_available_devices
from gravitorch.utils.device_placement import (
    CpuDevicePlacement,
    CudaDevicePlacement,
    ManualDevicePlacement,
    MpsDevicePlacement,
)

###########################################
#     Tests for ManualDevicePlacement     #
###########################################


def test_manual_device_placement_str() -> None:
    assert str(ManualDevicePlacement(torch.device("cpu"))).startswith("ManualDevicePlacement(")


def test_manual_device_placement_device_object() -> None:
    assert ManualDevicePlacement(torch.device("cpu")).device == torch.device("cpu")


def test_manual_device_placement_device_str() -> None:
    assert ManualDevicePlacement("cpu").device == torch.device("cpu")


@mark.parametrize("device", get_available_devices())
def test_manual_device_placement_send(device: str) -> None:
    device_placement = ManualDevicePlacement(torch.device(device))
    assert device_placement.send(torch.ones(2, 3)).equal(torch.ones(2, 3, device=device))


########################################
#     Tests for CpuDevicePlacement     #
########################################


def test_cpu_device_placement_device() -> None:
    assert CpuDevicePlacement().device == torch.device("cpu")


#########################################
#     Tests for CudaDevicePlacement     #
#########################################


def test_cuda_device_placement_device_default() -> None:
    assert CudaDevicePlacement().device == torch.device("cuda:0")


@mark.parametrize("index", (0, 1))
def test_cuda_device_placement_device_index(index: int) -> None:
    assert CudaDevicePlacement(index).device == torch.device(f"cuda:{index}")


########################################
#     Tests for MpsDevicePlacement     #
########################################


def test_mps_device_placement_device() -> None:
    assert MpsDevicePlacement().device == torch.device("mps")
