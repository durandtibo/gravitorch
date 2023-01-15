import torch
from pytest import mark
from torch import nn

from gravitorch.nn import FusionFFN, SumFusion
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


###############################
#     Tests for FusionFFN     #
###############################


def test_fusion_ffn_modules():
    module = FusionFFN(fusion=SumFusion(), ffn=nn.Linear(4, 7))
    assert isinstance(module.fusion, SumFusion)
    assert isinstance(module.ffn, nn.Linear)


@mark.parametrize("output_size", SIZES)
def test_fusion_ffn_output_size(output_size: int):
    assert FusionFFN(fusion=SumFusion(), ffn=nn.Linear(4, output_size)).output_size == output_size


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_fusion_ffn_forward(device: str, batch_size: int, input_size: int, output_size: int):
    device = torch.device(device)
    net = FusionFFN(fusion=SumFusion(), ffn=nn.Linear(input_size, output_size)).to(device=device)
    y = net(
        torch.ones(batch_size, input_size, device=device),
        3 * torch.ones(batch_size, input_size, device=device),
    )
    assert y.shape == (batch_size, output_size)
    assert y.dtype == torch.float
    assert y.device == device
