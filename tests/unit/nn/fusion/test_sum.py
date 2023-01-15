import torch
from pytest import mark, raises

from gravitorch.nn import AverageFusion, SumFusion
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


###############################
#     Tests for SumFusion     #
###############################


def test_sum_fusion_str():
    assert str(SumFusion()).startswith("SumFusion(")


def test_sum_fusion_0_input():
    module = SumFusion()
    with raises(ValueError):
        module()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_1_input(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_2_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(4))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_3_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(2),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(6))


@mark.parametrize("device", get_available_devices())
def test_sum_fusion_backward(device: str):
    device = torch.device(device)
    module = SumFusion().to(device=device)
    y = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    y.mean().backward()
    assert y.equal(torch.ones(2, 4, device=device).mul(3))


###################################
#     Tests for AverageFusion     #
###################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_1_input(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_2_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(2))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_3_inputs(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(2),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(2))


@mark.parametrize("device", get_available_devices())
def test_average_fusion_backward(device: str):
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    y = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    y.mean().backward()
    assert y.equal(torch.ones(2, 4, device=device))
