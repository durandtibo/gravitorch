from typing import Optional
from unittest.mock import patch

import torch
from pytest import mark

from gravitorch.nn.functional import asinh_barron_robust_loss, barron_robust_loss
from gravitorch.utils import get_available_devices

DTYPES = (torch.long, torch.float)
SIZES = (1, 2)


########################################
#     Tests for barron_robust_loss     #
########################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_1d(device: str, batch_size: int, alpha: float) -> None:
    device = torch.device(device)
    out = barron_robust_loss(
        prediction=torch.randn(batch_size, dtype=torch.float, device=device, requires_grad=True),
        target=torch.randn(batch_size, dtype=torch.float, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_barron_robust_loss_2d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = barron_robust_loss(
        prediction=torch.randn(
            batch_size, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
        alpha=alpha,
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
def test_barron_robust_loss_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = barron_robust_loss(
        prediction=torch.randn(
            batch_size, seq_len, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, seq_len, feature_size, dtype=torch.float, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


def test_barron_robust_loss_alpha_2_scale_1() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=2.0,
        scale=1.0,
    ).equal(torch.tensor(4.0))


def test_barron_robust_loss_alpha_2_scale_2() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=2.0,
        scale=2.0,
    ).equal(torch.tensor(1.0))


def test_barron_robust_loss_alpha_1_scale_1() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=1.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.2360679774997898),
    )


def test_barron_robust_loss_alpha_1_scale_2() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=1.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.41421356237309515),
    )


def test_barron_robust_loss_alpha_0_scale_1() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=0.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.0986122886681098),
    )


def test_barron_robust_loss_alpha_0_scale_2() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=0.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.4054651081081644),
    )


def test_barron_robust_loss_alpha_minus_2_scale_1() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=-2.0,
        scale=1.0,
    ).allclose(
        torch.tensor(1.0),
    )


def test_barron_robust_loss_alpha_minus_2_scale_2() -> None:
    assert barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
        alpha=-2.0,
        scale=2.0,
    ).allclose(
        torch.tensor(0.4),
    )


def test_barron_robust_loss_reduction_mean() -> None:
    assert barron_robust_loss(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="mean",
    ).allclose(
        torch.tensor(1 / 3),
    )


def test_barron_robust_loss_reduction_sum() -> None:
    assert barron_robust_loss(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="sum",
    ).equal(torch.tensor(2.0))


def test_barron_robust_loss_reduction_none() -> None:
    assert barron_robust_loss(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))


def test_barron_robust_loss_max_value() -> None:
    assert barron_robust_loss(
        prediction=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        max_value=0.5,
        reduction="none",
    ).equal(torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))


##############################################
#     Tests for asinh_barron_robust_loss     #
##############################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_asinh_barron_robust_loss_1d(device: str, batch_size: int, alpha: float) -> None:
    device = torch.device(device)
    out = asinh_barron_robust_loss(
        prediction=torch.randn(batch_size, dtype=torch.float, device=device, requires_grad=True),
        target=torch.randn(batch_size, dtype=torch.float, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
@mark.parametrize("alpha", (0, 1, 2))
def test_asinh_barron_robust_loss_2d(
    device: str, batch_size: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = asinh_barron_robust_loss(
        prediction=torch.randn(
            batch_size, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
        alpha=alpha,
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
def test_asinh_barron_robust_loss_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int, alpha: float
) -> None:
    device = torch.device(device)
    out = asinh_barron_robust_loss(
        prediction=torch.randn(
            batch_size, seq_len, feature_size, dtype=torch.float, device=device, requires_grad=True
        ),
        target=torch.randn(batch_size, seq_len, feature_size, dtype=torch.float, device=device),
        alpha=alpha,
    )
    out.backward()
    assert out.numel() == 1
    assert out.dtype == torch.float
    assert out.device == device


def test_asinh_barron_robust_loss_correct() -> None:
    assert asinh_barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=torch.ones(2, 3, dtype=torch.float),
    ).equal(torch.tensor(0.0))


def test_asinh_barron_robust_loss_partially_correct() -> None:
    assert asinh_barron_robust_loss(
        prediction=torch.ones(2, 2, dtype=torch.float),
        target=torch.tensor([[1, -1], [-1, 1]], dtype=torch.float),
    ).allclose(torch.tensor(1.553638799791392))


def test_asinh_barron_robust_loss_incorrect() -> None:
    assert asinh_barron_robust_loss(
        prediction=torch.ones(2, 3, dtype=torch.float),
        target=-torch.ones(2, 3, dtype=torch.float),
    ).allclose(torch.tensor(3.107277599582784))


@mark.parametrize("alpha", (1.0, 2.0))
@mark.parametrize("scale", (1.0, 2.0))
@mark.parametrize("max_value", (None, 1.0))
@mark.parametrize("reduction", ("mean", "sum"))
def test_asinh_barron_robust_loss_mock(
    alpha: float, scale: float, max_value: Optional[float], reduction: str
) -> None:
    with patch("gravitorch.nn.functional.barron_loss.barron_robust_loss") as loss_mock:
        asinh_barron_robust_loss(
            prediction=torch.tensor([0.0]),
            target=torch.tensor([0.0]),
            alpha=alpha,
            scale=scale,
            max_value=max_value,
            reduction=reduction,
        )
        loss_mock.assert_called_once_with(
            prediction=torch.tensor([0.0]),
            target=torch.tensor([0.0]),
            alpha=alpha,
            scale=scale,
            max_value=max_value,
            reduction=reduction,
        )
