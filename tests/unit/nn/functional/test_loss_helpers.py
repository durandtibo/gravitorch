import torch
from pytest import mark, raises

from gravitorch.nn.functional.loss_helpers import (
    basic_loss_reduction,
    check_basic_loss_reduction,
)
from gravitorch.utils import get_available_devices

DTYPES = (torch.long, torch.float)


##########################################
#     Tests for basic_loss_reduction     #
##########################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", DTYPES)
def test_basic_loss_reduction_mean(device: str, dtype: torch.dtype)-> None:
    device = torch.device(device)
    assert basic_loss_reduction(
        torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="mean"
    ).equal(torch.tensor(1.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", DTYPES)
def test_basic_loss_reduction_sum(device: str, dtype: torch.dtype)-> None:
    device = torch.device(device)
    assert basic_loss_reduction(
        torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="sum"
    ).equal(torch.tensor(6.0, dtype=dtype, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", DTYPES)
def test_basic_loss_reduction_none(device: str, dtype: torch.dtype)-> None:
    device = torch.device(device)
    assert basic_loss_reduction(
        torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device), reduction="none"
    ).equal(torch.tensor([[3, 2, 1], [1, 0, -1]], dtype=dtype, device=device))


def test_basic_loss_reduction_reduction_incorrect() -> None:
    with raises(ValueError):
        basic_loss_reduction(torch.ones(2, 2), reduction="incorrect")


################################################
#     Tests for check_basic_loss_reduction     #
################################################


@mark.parametrize("reduction", ("none", "mean", "sum"))
def test_check_basic_loss_reduction_valid(reduction: str) -> None:
    check_basic_loss_reduction(reduction)


def test_check_basic_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        check_basic_loss_reduction("incorrect")
