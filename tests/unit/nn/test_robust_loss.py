from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark

from gravitorch.nn import (
    AsinhMSELoss,
    MSLELoss,
    RelativeMSELoss,
    RelativeSmoothL1Loss,
    SymlogMSELoss,
    SymmetricRelativeSmoothL1Loss,
)

##################################
#     Tests for AsinhMSELoss     #
##################################


def test_asinh_mse_loss_str():
    assert str(AsinhMSELoss()).startswith("AsinhMSELoss(")


def test_asinh_mse_loss_forward():
    criterion = AsinhMSELoss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
def test_asinh_mse_loss_forward_reduction(reduction: str):
    criterion = AsinhMSELoss(reduction)
    with patch("gravitorch.nn.robust_loss.asinh_mse_loss") as asinh_mse_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            asinh_mse_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3), reduction)
        )
        assert objects_are_equal(asinh_mse_loss.call_args.kwargs, {})


##############################
#     Tests for MSLELoss     #
##############################


def test_msle_loss_str():
    assert str(MSLELoss()).startswith("MSLELoss(")


def test_msle_loss_forward():
    criterion = MSLELoss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
def test_msle_loss_forward_reduction(reduction: str):
    criterion = MSLELoss(reduction)
    with patch("gravitorch.nn.robust_loss.msle_loss") as msle_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            msle_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3), reduction)
        )
        assert objects_are_equal(msle_loss.call_args.kwargs, {})


###################################
#     Tests for SymlogMSELoss     #
###################################


def test_symlog_mse_loss_str():
    assert str(SymlogMSELoss()).startswith("SymlogMSELoss(")


def test_symlog_mse_loss_forward():
    criterion = SymlogMSELoss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
def test_symlog_mse_loss_forward_reduction(reduction: str):
    criterion = SymlogMSELoss(reduction)
    with patch("gravitorch.nn.robust_loss.symlog_mse_loss") as symlog_mse_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            symlog_mse_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3), reduction)
        )
        assert objects_are_equal(symlog_mse_loss.call_args.kwargs, {})


#####################################
#     Tests for RelativeMSELoss     #
#####################################


def test_relative_mse_loss_str():
    assert str(RelativeMSELoss()).startswith("RelativeMSELoss(")


def test_relative_mse_loss_eps_default():
    assert RelativeMSELoss()._eps == 1e-8


def test_relative_mse_loss_forward():
    criterion = RelativeMSELoss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
@mark.parametrize("eps", (1e-5, 1e-8))
def test_relative_mse_loss_forward_reduction(reduction: str, eps: float):
    criterion = RelativeMSELoss(reduction=reduction, eps=eps)
    with patch("gravitorch.nn.robust_loss.relative_mse_loss") as relative_mse_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            relative_mse_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3))
        )
        assert objects_are_equal(
            relative_mse_loss.call_args.kwargs, {"reduction": reduction, "eps": eps}
        )


##########################################
#     Tests for RelativeSmoothL1Loss     #
##########################################


def test_relative_smooth_l1_loss_str():
    assert str(RelativeSmoothL1Loss()).startswith("RelativeSmoothL1Loss(")


def test_relative_smooth_l1_loss_beta_default():
    assert RelativeSmoothL1Loss()._beta == 1.0


def test_relative_smooth_l1_loss_eps_default():
    assert RelativeSmoothL1Loss()._eps == 1e-8


def test_relative_smooth_l1_loss_forward():
    criterion = RelativeSmoothL1Loss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
@mark.parametrize("beta", (1.0, 10.0))
@mark.parametrize("eps", (1e-5, 1e-8))
def test_relative_smooth_l1_loss_forward_reduction(reduction: str, beta: float, eps: float):
    criterion = RelativeSmoothL1Loss(reduction=reduction, beta=beta, eps=eps)
    with patch("gravitorch.nn.robust_loss.relative_smooth_l1_loss") as relative_smooth_l1_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            relative_smooth_l1_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3))
        )
        assert objects_are_equal(
            relative_smooth_l1_loss.call_args.kwargs,
            {"reduction": reduction, "beta": beta, "eps": eps},
        )


###################################################
#     Tests for SymmetricRelativeSmoothL1Loss     #
###################################################


def test_symmetric_relative_smooth_l1_loss_str():
    assert str(SymmetricRelativeSmoothL1Loss()).startswith("SymmetricRelativeSmoothL1Loss(")


def test_symmetric_relative_smooth_l1_loss_beta_default():
    assert SymmetricRelativeSmoothL1Loss()._beta == 1.0


def test_symmetric_relative_smooth_l1_loss_eps_default():
    assert SymmetricRelativeSmoothL1Loss()._eps == 1e-8


def test_symmetric_relative_smooth_l1_loss_forward():
    criterion = SymmetricRelativeSmoothL1Loss()
    loss = criterion(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3))
    loss.backward()
    assert loss.equal(torch.tensor(0.0))


@mark.parametrize("reduction", ("mean", "sum", "none"))
@mark.parametrize("beta", (1.0, 10.0))
@mark.parametrize("eps", (1e-5, 1e-8))
def test_symmetric_relative_smooth_l1_loss_forward_reduction(
    reduction: str, beta: float, eps: float
):
    criterion = SymmetricRelativeSmoothL1Loss(reduction=reduction, beta=beta, eps=eps)
    with patch(
        "gravitorch.nn.robust_loss.symmetric_relative_smooth_l1_loss"
    ) as sym_rel_smooth_l1_loss:
        criterion(torch.ones(2, 3), torch.zeros(2, 3))
        assert objects_are_equal(
            sym_rel_smooth_l1_loss.call_args.args, (torch.ones(2, 3), torch.zeros(2, 3))
        )
        assert objects_are_equal(
            sym_rel_smooth_l1_loss.call_args.kwargs,
            {"reduction": reduction, "beta": beta, "eps": eps},
        )
