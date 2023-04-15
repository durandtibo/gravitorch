from unittest.mock import patch

import torch
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.networks import BetaMLP
from gravitorch.models.utils import is_loss_decreasing_with_sgd
from gravitorch.nn.experimental import ExponentialNLLLoss
from gravitorch.nn.functional.loss_helpers import VALID_REDUCTIONS
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


########################################
#     Tests for ExponentialNLLLoss     #
########################################


def test_exponential_nll_loss_module_str() -> None:
    assert str(ExponentialNLLLoss())


@mark.parametrize("eps", (1e-4, 1))
def test_exponential_nll_loss_module_eps(eps: float) -> None:
    assert ExponentialNLLLoss(eps=eps)._eps == eps


def test_exponential_nll_loss_module_eps_default() -> None:
    assert ExponentialNLLLoss()._eps == 1e-8


def test_exponential_nll_loss_module_incorrect_eps() -> None:
    with raises(ValueError, match="eps has to be greater or equal to 0"):
        ExponentialNLLLoss(eps=-1)


@mark.parametrize("max_log_value", (1, 2))
def test_exponential_nll_loss_module_max_log_value(max_log_value: float) -> None:
    assert ExponentialNLLLoss(max_log_value=max_log_value)._max_log_value == max_log_value


def test_exponential_nll_loss_module_max_log_value_default() -> None:
    assert ExponentialNLLLoss()._max_log_value == 20.0


@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_exponential_nll_loss_module_reduction(reduction: str) -> None:
    assert ExponentialNLLLoss(reduction=reduction).reduction == reduction


def test_exponential_nll_loss_module_incorrect_reduction() -> None:
    with raises(ValueError, match="Incorrect reduction: incorrect."):
        ExponentialNLLLoss(reduction="incorrect")


def test_exponential_nll_loss_module_forward_log_input_true() -> None:
    criterion = ExponentialNLLLoss()
    assert criterion(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(3.767288658951674),
    )


def test_exponential_nll_loss_module_forward_log_input_false() -> None:
    criterion = ExponentialNLLLoss(log_input=False)
    assert criterion(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(7.242511187797475),
    )


def test_exponential_nll_loss_module_forward_reduction_sum() -> None:
    criterion = ExponentialNLLLoss(reduction="sum")
    assert criterion(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(22.603731953710042),
    )


def test_exponential_nll_loss_module_forward_reduction_none() -> None:
    criterion = ExponentialNLLLoss(reduction="none")
    assert criterion(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(
            [[0.0, 1.718281828459045, 12.7781121978613], [5.38905609893065, 1.718281828459045, 1.0]]
        ),
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_exponential_nll_loss_module_forward_2d(
    device: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = ExponentialNLLLoss()
    out = criterion(
        log_rate=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
        target=torch.rand(batch_size, feature_size, dtype=torch.float, device=device),
    )
    assert out.numel() == 1
    assert out.shape == ()
    assert out.dtype == torch.float
    assert out.device == device


def test_exponential_nll_loss_module_forward_large_values() -> None:
    criterion = ExponentialNLLLoss()
    out = criterion(
        log_rate=100 * torch.randn(2, 3, dtype=torch.float),
        target=torch.rand(2, 3, dtype=torch.float),
    )
    assert not torch.isnan(out)
    assert not torch.isinf(out)


def test_exponential_nll_loss_module_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=BetaMLP(input_size=6, hidden_sizes=(8, 1)),
            criterion=VanillaLoss(criterion=ExponentialNLLLoss()),
        ),
        batch={ct.INPUT: torch.randn(4, 6), ct.TARGET: torch.rand(4, 1)},
    )


@mark.parametrize("log_input", (True, False))
@mark.parametrize("eps", (1e-8, 1e-7))
@mark.parametrize("max_log_value", (10, 20))
@mark.parametrize("reduction", VALID_REDUCTIONS)
def test_exponential_nll_loss_module_forward_mock(
    log_input: bool, eps: float, max_log_value: float, reduction: str
) -> None:
    criterion = ExponentialNLLLoss(
        log_input=log_input,
        eps=eps,
        max_log_value=max_log_value,
        reduction=reduction,
    )
    log_rate = torch.rand(1)
    target = torch.rand(1)
    with patch("gravitorch.nn.experimental.exponential_nll.exponential_nll_loss") as loss_mock:
        criterion(log_rate, target)
        loss_mock.assert_called_once_with(
            log_rate=log_rate,
            target=target,
            log_input=log_input,
            eps=eps,
            max_log_value=max_log_value,
            reduction=reduction,
        )
