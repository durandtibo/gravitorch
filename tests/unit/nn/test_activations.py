import torch
from pytest import mark

from gravitorch.nn import ReLUn, SquaredReLU
from gravitorch.utils import get_available_devices

###########################
#     Tests for ReLUn     #
###########################


def test_relun_str():
    assert str(ReLUn()).startswith("ReLUn(")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", (torch.float, torch.long))
def test_relun_forward(device: str, dtype: torch.dtype):
    device = torch.device(device)
    module = ReLUn().to(device=device)
    assert module(torch.arange(-1, 4, dtype=dtype, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_relun_forward_max_value_2(device: str):
    device = torch.device(device)
    module = ReLUn(max_value=2).to(device=device)
    assert module(torch.arange(-1, 4, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device)
    )


#################################
#     Tests for SquaredReLU     #
#################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", (torch.float, torch.long))
def test_squared_relu_forward(device: str, dtype: torch.dtype):
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    assert module(torch.arange(-1, 4, dtype=dtype, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 4.0, 9.0], dtype=torch.float, device=device)
    )
