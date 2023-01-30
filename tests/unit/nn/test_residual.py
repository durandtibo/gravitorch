import torch
from pytest import mark
from torch.nn import Identity, Linear, ReLU, Sequential

from gravitorch.nn import ResidualBlock
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


###################################
#     Tests for ResidualBlock     #
###################################


def test_residual_block_residual():
    assert isinstance(ResidualBlock(residual=Linear(4, 4)).residual, Linear)


def test_residual_block_skip_default():
    assert isinstance(ResidualBlock(residual=Linear(4, 4)).skip, Identity)


def test_residual_block_skip():
    assert isinstance(ResidualBlock(residual=Linear(4, 4), skip=Linear(4, 4)).skip, Linear)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_residual_block_forward(device: str, batch_size: int, mode: bool):
    device = torch.device(device)
    module = ResidualBlock(residual=Linear(4, 4)).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_residual_block_forward_skip(device: str, batch_size: int, mode: bool):
    device = torch.device(device)
    module = ResidualBlock(
        residual=Sequential(Linear(4, 8), ReLU(), Linear(8, 4)), skip=Linear(4, 4)
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 4, device=device))
    assert out.shape == (batch_size, 4)
    assert out.device == device
    assert out.dtype == torch.float
