from typing import Optional
from unittest.mock import patch

import torch
from pytest import mark, raises

from gravitorch.nn import AsinhBarronRobustLoss, BarronRobustLoss
from gravitorch.nn.functional.loss_helpers import VALID_REDUCTIONS
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


######################################
#     Tests for BarronRobustLoss     #
######################################


def test_barron_robust_loss_str() -> None:
    assert str(BarronRobustLoss()).startswith("BarronRobustLoss(")


@mark.parametrize("alpha", (-1, 0, 1, 2))
def test_barron_robust_loss_alpha(alpha: float) -> None:
    assert BarronRobustLoss(alpha=alpha)._alpha == alpha


def test_barron_robust_loss_alpha_default() -> None:
    assert BarronRobustLoss()._alpha == 2.0


@mark.parametrize("scale", (1, 2))
def test_barron_robust_loss_scale(scale: float) -> None:
    assert BarronRobustLoss(scale=scale)._scale == scale


def test_barron_robust_loss_scale_default() -> None:
    assert BarronRobustLoss()._scale == 1.0


def test_barron_robust_loss_incorrect_scale() -> None:
    with raises(ValueError):
        BarronRobustLoss(scale=0)


@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_barron_robust_loss_reduction(reduction: str) -> None:
    assert BarronRobustLoss(reduction=reduction).reduction == reduction


def test_barron_robust_loss_reduction_default() -> None:
    assert BarronRobustLoss().reduction == "mean"


def test_barron_robust_loss_incorrect_reduction() -> None:
    with raises(ValueError):
        BarronRobustLoss(reduction="incorrect")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_forward_1d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = BarronRobustLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(batch_size, dtype=torch.float, device=device, requires_grad=True),
        target=torch.randn(batch_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_forward_2d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = BarronRobustLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(
            batch_size, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_forward_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    criterion = BarronRobustLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(
            batch_size, seq_len, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, seq_len, feature_size, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


def test_barron_robust_loss_forward_alpha_2_scale_1() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=1.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).equal(torch.tensor(4.0))


def test_barron_robust_loss_forward_alpha_2_scale_2() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=2.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).equal(torch.tensor(1.0))


def test_barron_robust_loss_forward_alpha_1_scale_1() -> None:
    criterion = BarronRobustLoss(alpha=1.0, scale=1.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(1.2360679774997898),
    )


def test_barron_robust_loss_forward_alpha_1_scale_2() -> None:
    criterion = BarronRobustLoss(alpha=1.0, scale=2.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(0.41421356237309515),
    )


def test_barron_robust_loss_forward_alpha_0_scale_1() -> None:
    criterion = BarronRobustLoss(alpha=0.0, scale=1.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(1.0986122886681098),
    )


def test_barron_robust_loss_forward_alpha_0_scale_2() -> None:
    criterion = BarronRobustLoss(alpha=0.0, scale=2.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(0.4054651081081644),
    )


def test_barron_robust_loss_forward_alpha_minus_2_scale_1() -> None:
    criterion = BarronRobustLoss(alpha=-2.0, scale=1.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(1.0),
    )


def test_barron_robust_loss_forward_alpha_minus_2_scale_2() -> None:
    criterion = BarronRobustLoss(alpha=-2.0, scale=2.0)
    assert criterion(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(
        torch.tensor(0.4),
    )


def test_barron_robust_loss_forward_reduction_mean() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=1.0, reduction="mean")
    assert criterion(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(1 / 3),
    )


def test_barron_robust_loss_forward_reduction_sum() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=1.0, reduction="sum")
    assert criterion(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).equal(torch.tensor(2.0))


def test_barron_robust_loss_forward_reduction_none() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=1.0, reduction="none")
    assert criterion(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))


def test_barron_robust_loss_forward_max_value() -> None:
    criterion = BarronRobustLoss(alpha=2.0, scale=1.0, max_value=0.5, reduction="none")
    assert criterion(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))


@mark.parametrize("alpha", (1.0, 2.0))
@mark.parametrize("scale", (1.0, 2.0))
@mark.parametrize("max_value", (None, 1.0))
@mark.parametrize("reduction", ("mean", "sum"))
def test_barron_robust_loss_forward_mock(
    alpha: float, scale: float, max_value: Optional[float], reduction: str
) -> None:
    criterion = BarronRobustLoss(alpha=alpha, scale=scale, max_value=max_value, reduction=reduction)
    with patch("gravitorch.nn.barron_loss.barron_robust_loss") as loss_mock:
        criterion(prediction=torch.tensor([1.0]), target=torch.tensor([1.0]))
        loss_mock.assert_called_once_with(
            prediction=torch.tensor([1.0]),
            target=torch.tensor([1.0]),
            alpha=alpha,
            scale=scale,
            max_value=max_value,
            reduction=reduction,
        )


###########################################
#     Tests for AsinhBarronRobustLoss     #
###########################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_forward(device: str, alpha: float) -> None:
    device = torch.device(device)
    criterion = BarronRobustLoss(alpha=alpha)
    out = criterion(
        prediction=torch.randn(2, 3, dtype=torch.float, device=device, requires_grad=True),
        target=torch.randn(2, 3, dtype=torch.float, device=device),
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("alpha", (1.0, 2.0))
@mark.parametrize("scale", (1.0, 2.0))
@mark.parametrize("max_value", (None, 1.0))
@mark.parametrize("reduction", ("mean", "sum"))
def test_asinh_barron_robust_loss_forward_mock(
    alpha: float, scale: float, max_value: Optional[float], reduction: str
) -> None:
    criterion = AsinhBarronRobustLoss(
        alpha=alpha, scale=scale, max_value=max_value, reduction=reduction
    )
    with patch("gravitorch.nn.barron_loss.asinh_barron_robust_loss") as loss_mock:
        criterion(prediction=torch.tensor([1.0]), target=torch.tensor([1.0]))
        loss_mock.assert_called_once_with(
            prediction=torch.tensor([1.0]),
            target=torch.tensor([1.0]),
            alpha=alpha,
            scale=scale,
            max_value=max_value,
            reduction=reduction,
        )
