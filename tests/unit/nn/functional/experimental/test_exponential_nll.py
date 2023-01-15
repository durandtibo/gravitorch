import torch
from pytest import mark, raises

from gravitorch.nn.functional.experimental import exponential_nll_loss
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


##########################################
#     Tests for exponential_nll_loss     #
##########################################


def test_exponential_nll_loss_log_input_true():
    assert exponential_nll_loss(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(3.767288658951674),
    )


def test_exponential_nll_loss_log_input_false():
    assert exponential_nll_loss(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        log_input=False,
    ).allclose(
        torch.tensor(7.242511187797475),
    )


def test_exponential_nll_loss_log_reduction_sum():
    assert exponential_nll_loss(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="sum",
    ).allclose(
        torch.tensor(22.603731953710042),
    )


def test_exponential_nll_loss_log_reduction_none():
    assert exponential_nll_loss(
        log_rate=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="none",
    ).allclose(
        torch.tensor(
            [[0.0, 1.718281828459045, 12.7781121978613], [5.38905609893065, 1.718281828459045, 1.0]]
        ),
    )


def test_exponential_nll_loss_log_incorrect_reduction():
    with raises(ValueError):
        exponential_nll_loss(
            log_rate=torch.tensor([0, 1, 2], dtype=torch.float),
            target=torch.tensor([0, 1, 2], dtype=torch.float),
            reduction="incorrect",
        )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_exponential_nll_loss_2d(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    out = exponential_nll_loss(
        log_rate=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
        target=torch.rand(batch_size, feature_size, dtype=torch.float, device=device),
    )
    assert out.numel() == 1
    assert out.shape == tuple()
    assert out.dtype == torch.float
    assert out.device == device


def test_exponential_nll_loss_large_values():
    out = exponential_nll_loss(
        log_rate=100 * torch.randn(2, 3, dtype=torch.float),
        target=torch.rand(2, 3, dtype=torch.float),
    )
    assert not torch.isnan(out)
    assert not torch.isinf(out)
