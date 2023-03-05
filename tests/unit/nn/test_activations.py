import torch
from pytest import mark

from gravitorch.nn import ReLUn, Snake, SquaredReLU
from gravitorch.utils import get_available_devices

SIZES = ((1, 1), (2, 3), (2, 3, 4), (2, 3, 4, 5))

###########################
#     Tests for ReLUn     #
###########################


def test_relun_str() -> None:
    assert str(ReLUn()).startswith("ReLUn(")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", (torch.float, torch.long))
def test_relun_forward(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    module = ReLUn().to(device=device)
    assert module(torch.arange(-1, 4, dtype=dtype, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_relun_forward_max_value_2(device: str) -> None:
    device = torch.device(device)
    module = ReLUn(max_value=2).to(device=device)
    assert module(torch.arange(-1, 4, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device)
    )


@mark.parametrize("size", SIZES)
def test_relun_forward_size(size: tuple[int, ...]) -> None:
    module = ReLUn()
    out = module(torch.randn(*size))
    assert out.shape == size
    assert out.dtype == torch.float


###########################
#     Tests for Snake     #
###########################


def test_snake_str() -> None:
    assert str(Snake()).startswith("Snake(")


def test_snake_forward_frequency_default() -> None:
    module = Snake()
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.708073418273571, 0.0, -0.2919265817264288],
                [-1.173178189568194, 0.0, 2.826821810431806],
            ]
        ),
    )


def test_snake_forward_frequency_2() -> None:
    module = Snake(frequency=2)
    assert torch.allclose(
        module(torch.tensor([[1.0, 0.0, -1.0], [-2.0, 0.0, 2.0]])),
        torch.tensor(
            [
                [1.413410905215903, 0.0, -0.586589094784097],
                [-1.7136249915478468, 0.0, 2.2863750084521532],
            ]
        ),
    )


@mark.parametrize("size", SIZES)
def test_snake_forward_size(size: tuple[int, ...]) -> None:
    module = Snake()
    out = module(torch.randn(*size))
    assert out.shape == size
    assert out.dtype == torch.float


#################################
#     Tests for SquaredReLU     #
#################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", (torch.float, torch.long))
def test_squared_relu_forward(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    module = SquaredReLU().to(device=device)
    assert module(torch.arange(-1, 4, dtype=dtype, device=device)).equal(
        torch.tensor([0.0, 0.0, 1.0, 4.0, 9.0], dtype=torch.float, device=device)
    )


@mark.parametrize("size", SIZES)
def test_squared_relu_forward_size(size: tuple[int, ...]) -> None:
    module = SquaredReLU()
    out = module(torch.randn(*size))
    assert out.shape == size
    assert out.dtype == torch.float
