from typing import Union

import torch
from objectory import OBJECT_TARGET
from pytest import mark
from torch.nn import GELU, Module, ReLU

from gravitorch.nn import ExU
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#########################
#     Tests for ExU     #
#########################


def test_exu_str():
    assert str(ExU(input_size=4, output_size=6)).startswith("ExU(")


@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_exu_input_output_sizes(input_size: int, output_size: int):
    module = ExU(input_size=input_size, output_size=output_size)
    assert module.input_size == input_size
    assert module.output_size == output_size
    assert module.weight.shape == (input_size, output_size)
    assert module.bias.shape == (input_size,)


def test_exu_activation_default():
    assert isinstance(ExU(input_size=4, output_size=6).activation, ReLU)


@mark.parametrize("activation", (GELU(), {OBJECT_TARGET: "torch.nn.GELU"}))
def test_exu_activation_gelu(activation: Union[Module, dict]):
    assert isinstance(ExU(input_size=4, output_size=6).activation, ReLU)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_exu_forward_2d(device: str, batch_size: int):
    device = torch.device(device)
    module = ExU(input_size=4, output_size=6).to(device=device)
    output = module(torch.randn(batch_size, 4, device=device))
    assert output.shape == (batch_size, 6)
    assert output.dtype == torch.float
    assert output.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_exu_forward_3d(device: str, batch_size: int):
    device = torch.device(device)
    module = ExU(input_size=4, output_size=6).to(device=device)
    output = module(torch.randn(batch_size, 3, 4, device=device))
    assert output.shape == (batch_size, 3, 6)
    assert output.dtype == torch.float
    assert output.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_exu_forward_4d(device: str, batch_size: int):
    device = torch.device(device)
    module = ExU(input_size=4, output_size=6).to(device=device)
    output = module(torch.randn(batch_size, 3, 4, 4, device=device))
    assert output.shape == (batch_size, 3, 4, 6)
    assert output.dtype == torch.float
    assert output.device == device
