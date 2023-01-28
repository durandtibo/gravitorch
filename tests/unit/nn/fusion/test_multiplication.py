import torch
from pytest import mark, raises

from gravitorch.nn import MultiplicationFusion
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


##########################################
#     Tests for MultiplicationFusion     #
##########################################


def test_multiplication_fusion_forward_0_input():
    module = MultiplicationFusion()
    with raises(ValueError, match="must have at least one tensor in the input"):
        module()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_1_input(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_2_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(0.25))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_3_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(0.125))


@mark.parametrize("device", get_available_devices())
def test_multiplication_fusion_backward(device: str):
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    y = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    y.mean().backward()
    assert y.equal(torch.ones(2, 4, device=device))
