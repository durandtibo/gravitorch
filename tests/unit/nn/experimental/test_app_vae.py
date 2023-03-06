from unittest.mock import patch

import torch
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.networks import BetaMLP
from gravitorch.models.utils import is_loss_decreasing_with_sgd
from gravitorch.nn.experimental import AppVAETimeLoss
from gravitorch.nn.functional.loss_helpers import VALID_REDUCTIONS
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


####################################
#     Tests for AppVAETimeLoss     #
####################################


def test_app_vae_time_loss_module_str() -> None:
    assert str(AppVAETimeLoss())


@mark.parametrize("delta", (0.01, 1))
def test_app_vae_time_loss_module_delta(delta: float) -> None:
    assert AppVAETimeLoss(delta=delta)._delta == delta


def test_app_vae_time_loss_module_delta_default() -> None:
    assert AppVAETimeLoss()._delta == 0.1


def test_app_vae_time_loss_module_incorrect_delta() -> None:
    with raises(ValueError):
        AppVAETimeLoss(delta=-1)


@mark.parametrize("eps", (1e-4, 1))
def test_app_vae_time_loss_module_eps(eps: float) -> None:
    assert AppVAETimeLoss(eps=eps)._eps == eps


def test_app_vae_time_loss_module_eps_default() -> None:
    assert AppVAETimeLoss()._eps == 1e-8


def test_app_vae_time_loss_module_incorrect_eps() -> None:
    with raises(ValueError):
        AppVAETimeLoss(eps=-1)


@mark.parametrize("max_log_value", (1, 2))
def test_app_vae_time_loss_module_max_log_value(max_log_value: float) -> None:
    assert AppVAETimeLoss(max_log_value=max_log_value)._max_log_value == max_log_value


def test_app_vae_time_loss_module_max_log_value_default() -> None:
    assert AppVAETimeLoss()._max_log_value == 20.0


@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_app_vae_time_loss_module_reduction(reduction: str) -> None:
    assert AppVAETimeLoss(reduction=reduction).reduction == reduction


def test_app_vae_time_loss_module_incorrect_reduction() -> None:
    with raises(ValueError):
        AppVAETimeLoss(reduction="incorrect")


def test_app_vae_time_loss_module_forward_log_input_true() -> None:
    criterion = AppVAETimeLoss(log_input=True)
    assert criterion(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(6.246282676267209),
    )


def test_app_vae_time_loss_module_forward_reduction_mean() -> None:
    criterion = AppVAETimeLoss(log_input=False)
    assert criterion(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(8.826873668655658),
    )


def test_app_vae_time_loss_module_forward_reduction_sum() -> None:
    criterion = AppVAETimeLoss(reduction="sum")
    assert criterion(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(52.96124201193395),
    )


def test_app_vae_time_loss_module_forward_reduction_none() -> None:
    criterion = AppVAETimeLoss(reduction="none")
    assert criterion(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(
            [
                [18.420680743952367, 3.3521684610440903, 5.707771800970519],
                [3.7077718009705194, 3.3521684610440903, 18.420680743952367],
            ]
        ),
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_app_vae_time_loss_module_forward_2d(
    device: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = AppVAETimeLoss()
    out = criterion(
        lmbda=torch.rand(batch_size, feature_size, dtype=torch.float, device=device),
        target=torch.rand(batch_size, feature_size, dtype=torch.float, device=device),
    )
    assert out.numel() == 1
    assert out.shape == ()
    assert out.dtype == torch.float
    assert out.device == device


def test_app_vae_time_loss_module_forward_large_values() -> None:
    criterion = AppVAETimeLoss()
    out = criterion(
        lmbda=100 * torch.rand(2, 3, dtype=torch.float),
        target=torch.rand(2, 3, dtype=torch.float),
    )
    assert not torch.isnan(out)
    assert not torch.isinf(out)


def test_app_vae_time_loss_module_is_loss_decreasing() -> None:
    # Use log_lambda because the network output can be negative
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=BetaMLP(input_size=6, hidden_sizes=(8, 1)),
            criterion=VanillaLoss(criterion=AppVAETimeLoss(log_input=True)),
        ),
        batch={ct.INPUT: torch.rand(4, 6), ct.TARGET: torch.rand(4, 1)},
    )


@mark.parametrize("log_input", (True, False))
@mark.parametrize("delta", (0.1, 0.2))
@mark.parametrize("eps", (1e-8, 1e-7))
@mark.parametrize("max_log_value", (10, 20))
@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_app_vae_time_loss_module_forward_mock(
    log_input: bool,
    delta: float,
    eps: float,
    max_log_value: float,
    reduction: str,
) -> None:
    criterion = AppVAETimeLoss(
        log_input=log_input,
        delta=delta,
        eps=eps,
        max_log_value=max_log_value,
        reduction=reduction,
    )
    lmbda = torch.rand(1)
    target = torch.rand(1)
    with patch("gravitorch.nn.experimental.app_vae.app_vae_time_loss") as loss_mock:
        criterion(lmbda, target)
        loss_mock.assert_called_once_with(
            lmbda=lmbda,
            target=target,
            log_input=log_input,
            delta=delta,
            eps=eps,
            max_log_value=max_log_value,
            reduction=reduction,
        )
