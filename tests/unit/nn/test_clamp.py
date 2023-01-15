from typing import Union

import torch
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.nn import Module, MSELoss, SmoothL1Loss

from gravitorch.nn import Clamp, ClampLoss
from gravitorch.nn.functional.loss_helpers import VALID_REDUCTIONS
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


###########################
#     Tests for Clamp     #
###########################


def test_clamp_str():
    assert str(Clamp()).startswith("Clamp(")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", (torch.float, torch.long))
def test_clamp_forward(device: str, dtype: torch.dtype):
    device = torch.device(device)
    module = Clamp().to(device=device)
    assert module(torch.arange(-3, 4, dtype=dtype, device=device)).equal(
        torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_clamp_forward_min_value_0(device: str):
    device = torch.device(device)
    module = Clamp(min_value=0.0).to(device=device)
    assert module(torch.arange(-3, 4, device=device)).equal(
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_clamp_forward_max_value_2(device: str):
    device = torch.device(device)
    module = Clamp(max_value=2).to(device=device)
    assert module(torch.arange(-3, 4, device=device)).equal(
        torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float, device=device)
    )


###############################
#     Tests for ClampLoss     #
###############################


@mark.parametrize(
    "criterion,criterion_cls",
    (
        (MSELoss(), MSELoss),
        (SmoothL1Loss(), SmoothL1Loss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, MSELoss),
    ),
)
def test_clamp_loss_criterion(criterion: Union[Module, dict], criterion_cls: type[Module]):
    assert isinstance(ClampLoss(criterion, min_value=0.0, max_value=1.0).criterion, criterion_cls)


@mark.parametrize("min_value", (1, 2))
def test_clamp_loss_min_value(min_value: float):
    assert ClampLoss(MSELoss(), min_value=min_value, max_value=None).clamp._min_value == min_value


@mark.parametrize("max_value", (1, 2))
def test_clamp_loss_max_value(max_value: float):
    assert ClampLoss(MSELoss(), min_value=None, max_value=max_value).clamp._max_value == max_value


@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_clamp_loss_reduction(reduction: str):
    assert (
        ClampLoss(MSELoss(), min_value=0.0, max_value=1.0, reduction=reduction).reduction
        == reduction
    )


def test_clamp_loss_reduction_default():
    assert ClampLoss(MSELoss(), min_value=0.0, max_value=1.0).reduction == "none"


def test_clamp_loss_incorrect_reduction():
    with raises(ValueError):
        ClampLoss(MSELoss(), min_value=0.0, max_value=1.0, reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("max_value", (1, 2))
def test_clamp_loss_forward_mse_max_value(
    device: str, batch_size: int, feature_size: int, max_value: float
):
    device = torch.device(device)
    criterion = ClampLoss(MSELoss(), min_value=None, max_value=max_value).to(device=device)
    assert criterion(
        torch.full((batch_size, feature_size), 2, device=device),
        torch.zeros(batch_size, feature_size, device=device),
    ).equal(torch.tensor(max_value, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("min_value", (1, 2))
def test_clamp_loss_forward_mse_min_value(
    device: str, batch_size: int, feature_size: int, min_value: float
):
    device = torch.device(device)
    criterion = ClampLoss(MSELoss(), min_value=min_value, max_value=None).to(device=device)
    assert criterion(
        torch.ones(batch_size, feature_size, device=device),
        torch.ones(batch_size, feature_size, device=device),
    ).equal(torch.tensor(min_value, dtype=torch.float, device=device))


def test_clamp_loss_forward_mse_reduction_mean():
    criterion = ClampLoss(MSELoss(reduction="none"), min_value=0.0, max_value=2.0, reduction="mean")
    assert criterion(
        torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[0, 1, 2], [4, 1, 0]], dtype=torch.float),
    ).equal(torch.tensor(0.5))


def test_clamp_loss_forward_mse_reduction_sum():
    criterion = ClampLoss(MSELoss(reduction="none"), min_value=0.0, max_value=2.0, reduction="sum")
    assert criterion(
        torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[0, 1, 2], [4, 1, 0]], dtype=torch.float),
    ).equal(torch.tensor(3.0))


def test_clamp_loss_forward_mse_reduction_none():
    criterion = ClampLoss(
        MSELoss(reduction="none"), min_value=None, max_value=2.0, reduction="none"
    )
    assert criterion(
        torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        torch.tensor([[0, 1, 2], [4, 1, 0]], dtype=torch.float),
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]]))
