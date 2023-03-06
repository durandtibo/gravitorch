import torch
from pytest import mark

from gravitorch.nn import FusionNorm
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


################################
#     Tests for FusionNorm     #
################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_fusion_norm_sum_layer_norm(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    net = FusionNorm.create_sum_layer_norm(input_size=input_size).to(device=device)
    y = net(
        torch.ones(batch_size, input_size, device=device),
        3 * torch.ones(batch_size, input_size, device=device),
    )
    assert y.shape == (batch_size, input_size)
    assert y.dtype == torch.float
    assert y.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_fusion_norm_multiplication_layer_norm(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    net = FusionNorm.create_multiplication_layer_norm(input_size=input_size).to(device=device)
    y = net(
        torch.ones(batch_size, input_size, device=device),
        3 * torch.ones(batch_size, input_size, device=device),
    )
    assert y.shape == (batch_size, input_size)
    assert y.dtype == torch.float
    assert y.device == device
