import torch
from pytest import mark, raises
from torch import nn

from gravitorch.nn.functional import (
    asinh_mse_loss,
    log_cosh_loss,
    msle_loss,
    relative_mse_loss,
    relative_smooth_l1_loss,
    symlog_mse_loss,
    symmetric_relative_smooth_l1_loss,
)
from gravitorch.nn.utils import is_loss_decreasing_with_sgd
from gravitorch.utils import get_available_devices

DTYPES = (torch.long, torch.float)
SIZES = (1, 2)


###############################
#     Tests for msle_loss     #
###############################


@mark.parametrize("device", get_available_devices())
def test_msle_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_msle_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 3, device=device), torch.zeros(2, 3, device=device)).allclose(
        torch.tensor(0.4804530139182014, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_msle_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(2, 2, device=device), torch.eye(2, device=device)).allclose(
        torch.tensor(0.2402265069591007, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_msle_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(
        torch.ones(2, 2, device=device), torch.eye(2, device=device), reduction="sum"
    ).allclose(torch.tensor(0.9609060278364028, device=device))


@mark.parametrize("device", get_available_devices())
def test_msle_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert msle_loss(
        torch.ones(2, 2, device=device), torch.eye(2, device=device), reduction="none"
    ).allclose(
        torch.tensor([[0.0, 0.4804530139182014], [0.4804530139182014, 0.0]], device=device),
    )


def test_msle_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        msle_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_msle_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert msle_loss(torch.ones(*shape, device=device), torch.ones(*shape, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


def test_msle_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Sequential(nn.Linear(4, 2), nn.Sigmoid()),
        criterion=msle_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


####################################
#     Tests for asinh_mse_loss     #
####################################


@mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(3.107277599582784, device=device))


@mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1
    ).allclose(torch.tensor(1.553638799791392, device=device))


@mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="sum",
    ).allclose(torch.tensor(6.214555199165568, device=device))


@mark.parametrize("device", get_available_devices())
def test_asinh_mse_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="none",
    ).allclose(
        torch.tensor([[0.0, 3.107277599582784], [3.107277599582784, 0.0]], device=device),
    )


def test_asinh_mse_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        asinh_mse_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_asinh_mse_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert asinh_mse_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


def test_asinh_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=asinh_mse_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


#####################################
#     Tests for symlog_mse_loss     #
#####################################


@mark.parametrize("device", get_available_devices())
def test_symlog_mse_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_symlog_mse_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(1.9218120556728056, device=device))


@mark.parametrize("device", get_available_devices())
def test_symlog_mse_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(
        torch.ones(2, 2, device=device), 2 * torch.eye(2, device=device) - 1
    ).allclose(torch.tensor(0.9609060278364028, device=device))


@mark.parametrize("device", get_available_devices())
def test_symlog_mse_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="sum",
    ).allclose(torch.tensor(3.843624111345611, device=device))


@mark.parametrize("device", get_available_devices())
def test_symlog_mse_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(
        torch.ones(2, 2, device=device),
        2 * torch.eye(2, device=device) - 1,
        reduction="none",
    ).allclose(
        torch.tensor([[0.0, 1.9218120556728056], [1.9218120556728056, 0.0]], device=device),
    )


def test_symlog_mse_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        symlog_mse_loss(torch.ones(2, 2), torch.eye(2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_symlog_mse_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert symlog_mse_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


def test_symlog_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=symlog_mse_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


#######################################
#     Tests for relative_mse_loss     #
#######################################


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_correct_zeros(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).equal(torch.tensor(4.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.eye(2, device=device).mul(2).sub(1), torch.ones(2, 2, device=device)
    ).equal(torch.tensor(2.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.eye(2, device=device).mul(2).sub(1),
        torch.ones(2, 2, device=device),
        reduction="sum",
    ).equal(torch.tensor(8.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_mse_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.eye(2, device=device).mul(2).sub(1),
        torch.ones(2, 2, device=device),
        reduction="none",
    ).equal(torch.tensor([[0.0, 4.0], [4.0, 0.0]], device=device))


def test_relative_mse_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        relative_mse_loss(torch.ones(2, 2), torch.ones(2, 2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_relative_mse_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert relative_mse_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


def test_relative_mse_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=relative_mse_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


#############################################
#     Tests for relative_smooth_l1_loss     #
#############################################


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_correct_zeros(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).equal(torch.tensor(1.5, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device)
    ).equal(torch.tensor(0.25, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="sum",
    ).equal(torch.tensor(1.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_relative_smooth_l1_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.5], [0.5, 0.0]], device=device))


def test_relative_smooth_l1_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        relative_smooth_l1_loss(torch.ones(2, 2), torch.ones(2, 2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_relative_smooth_l1_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert relative_smooth_l1_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


def test_relative_smooth_l1_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=relative_smooth_l1_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


#######################################################
#     Tests for symmetric_relative_smooth_l1_loss     #
#######################################################


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_correct_zeros(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).equal(torch.tensor(1.5, device=device))


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device)
    ).equal(torch.tensor(0.5, device=device))


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="sum",
    ).equal(torch.tensor(2.0, device=device))


@mark.parametrize("device", get_available_devices())
def test_symmetric_relative_smooth_l1_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="none",
    ).equal(torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device))


def test_symmetric_relative_smooth_l1_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        symmetric_relative_smooth_l1_loss(torch.ones(2, 2), torch.ones(2, 2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_symmetric_relative_smooth_l1_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert symmetric_relative_smooth_l1_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


def test_symmetric_relative_smooth_l1_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=symmetric_relative_smooth_l1_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


###################################
#     Tests for log_cosh_loss     #
###################################


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_correct(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.ones(2, 3, device=device), torch.ones(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_correct_zeros(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.zeros(2, 3, device=device), torch.zeros(2, 3, device=device)).equal(
        torch.tensor(0.0, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_incorrect(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(2, 3, device=device), -torch.ones(2, 3, device=device)
    ).allclose(torch.tensor(1.3250027473578645, device=device))


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_partially_correct(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(torch.eye(2, device=device), torch.ones(2, 2, device=device)).allclose(
        torch.tensor(0.21689041524151356, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_scale_0_5(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device), scale=0.5
    ).allclose(torch.tensor(0.6625013736789322, device=device))


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_scale_2(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device), torch.ones(2, 2, device=device), scale=2.0
    ).allclose(torch.tensor(0.06005725347913873, device=device))


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_reduction_sum(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="sum",
    ).allclose(torch.tensor(0.8675616609660542, device=device))


@mark.parametrize("device", get_available_devices())
def test_log_cosh_loss_reduction_none(device: str) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.eye(2, device=device),
        torch.ones(2, 2, device=device),
        reduction="none",
    ).allclose(torch.tensor([[0.0, 0.4337808304830271], [0.4337808304830271, 0.0]], device=device))


def test_log_cosh_loss_reduction_incorrect() -> None:
    with raises(ValueError):
        log_cosh_loss(torch.ones(2, 2), torch.ones(2, 2), reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("shape", ((2,), (2, 3), (2, 3, 4)))
def test_log_cosh_loss_shape(device: str, shape: tuple[int, ...]) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(*shape, device=device), torch.ones(*shape, device=device)
    ).equal(torch.tensor(0.0, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_log_cosh_loss_dtype(device: str, dtype_prediction: torch.dtype, dtype_target: torch.dtype) -> None:
    device = torch.device(device)
    assert log_cosh_loss(
        torch.ones(2, 3, device=device, dtype=dtype_prediction),
        torch.ones(2, 3, device=device, dtype=dtype_target),
    ).equal(torch.tensor(0.0, device=device, dtype=torch.float))


def test_log_cosh_loss_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=log_cosh_loss,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
