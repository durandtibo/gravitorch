import torch
from pytest import mark, raises

from gravitorch.nn.functional.experimental import app_vae_time_loss
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#######################################
#     Tests for app_vae_time_loss     #
#######################################


def test_app_vae_time_loss_log_input_true():
    assert app_vae_time_loss(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        log_input=True,
    ).allclose(
        torch.tensor(6.246282676267209),
    )


def test_app_vae_time_loss_reduction_mean():
    assert app_vae_time_loss(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
    ).allclose(
        torch.tensor(8.826873668655658),
    )


def test_app_vae_time_loss_reduction_sum():
    assert app_vae_time_loss(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="sum",
    ).allclose(
        torch.tensor(52.96124201193395),
    )


def test_app_vae_time_loss_reduction_none():
    assert app_vae_time_loss(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        reduction="none",
    ).allclose(
        torch.tensor(
            [
                [18.420680743952367, 3.3521684610440903, 5.707771800970519],
                [3.7077718009705194, 3.3521684610440903, 18.420680743952367],
            ]
        ),
    )


def test_app_vae_time_loss_incorrect_reduction():
    with raises(ValueError):
        app_vae_time_loss(
            lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
            reduction="incorrect",
        )


def test_app_vae_time_loss_delta_1():
    assert app_vae_time_loss(
        lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
        target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
        delta=1.0,
    ).allclose(
        torch.tensor(7.674923115736104),
    )


def test_app_vae_time_loss_incorrect_delta():
    with raises(ValueError):
        app_vae_time_loss(
            lmbda=torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.float),
            target=torch.tensor([[0, 1, 2], [1, 1, 1]], dtype=torch.float),
            delta=0.0,
        )


def test_app_vae_time_loss_eps_1():
    assert app_vae_time_loss(
        lmbda=torch.tensor(0.0, dtype=torch.float),
        target=torch.tensor(0.0, dtype=torch.float),
        eps=1.0,
    ).allclose(
        torch.tensor(0.0),
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_app_vae_time_loss_2d(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    out = app_vae_time_loss(
        lmbda=torch.randn(batch_size, feature_size, dtype=torch.float, device=device),
        target=torch.rand(batch_size, feature_size, dtype=torch.float, device=device),
    )
    assert out.numel() == 1
    assert out.shape == tuple()
    assert out.dtype == torch.float
    assert out.device == device


def test_app_vae_time_loss_large_values():
    out = app_vae_time_loss(
        lmbda=100 * torch.rand(2, 3, dtype=torch.float),
        target=torch.rand(2, 3, dtype=torch.float),
    )
    assert not torch.isnan(out)
    assert not torch.isinf(out)
