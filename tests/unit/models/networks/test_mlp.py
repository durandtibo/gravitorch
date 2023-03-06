from collections.abc import Sequence

import torch
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.networks.mlp import (
    AlphaMLP,
    BetaMLP,
    create_alpha_mlp,
    create_beta_mlp,
)
from gravitorch.models.utils import is_loss_decreasing_with_sgd
from gravitorch.utils import get_available_devices

SIZES = (1, 2)
DROPOUT_VALUES = (0.1, 0.5)


##############################
#     Tests for AlphaMLP     #
##############################


@mark.parametrize("input_size", SIZES)
def test_alpha_mlp_input_size(input_size: int) -> None:
    net = AlphaMLP(input_size=input_size, hidden_sizes=(16, 4))
    assert net.input_size == input_size
    assert net.layers.linear1.in_features == input_size


@mark.parametrize("hidden_sizes,num_layers", (((8,), 2), ((16, 4), 4), ((16, 16, 4), 6)))
def test_alpha_mlp_num_layers(hidden_sizes: Sequence[int], num_layers: int) -> None:
    assert len(AlphaMLP(input_size=16, hidden_sizes=hidden_sizes)) == num_layers


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_alpha_mlp_get_dummy_input(device: str, batch_size: int, input_size: int):
    device = torch.device(device)
    net = AlphaMLP(input_size=input_size, hidden_sizes=(16, 4)).to(device=device)
    dummy_input = net.get_dummy_input(batch_size)
    assert isinstance(dummy_input, tuple)
    assert len(dummy_input) == 1
    assert dummy_input[0].shape == (batch_size, input_size)
    assert dummy_input[0].dtype == torch.float
    assert dummy_input[0].device == device


@mark.parametrize("input_name", ("name", ct.INPUT))
def test_alpha_mlp_get_input_names(input_name: str) -> None:
    net = AlphaMLP(input_size=16, hidden_sizes=(16, 4), input_name=input_name)
    assert net.get_input_names() == (input_name,)


@mark.parametrize("input_name", ("name", ct.INPUT))
@mark.parametrize("output_name", ("name", ct.PREDICTION))
def test_alpha_mlp_get_onnx_dynamic_axis(input_name: str, output_name: str) -> None:
    net = AlphaMLP(
        input_size=16, hidden_sizes=(16, 4), input_name=input_name, output_name=output_name
    )
    assert net.get_onnx_dynamic_axis() == {input_name: {0: "batch"}, output_name: {0: "batch"}}


@mark.parametrize("output_name", ("name", ct.PREDICTION))
def test_alpha_mlp_get_output_names(output_name: str) -> None:
    net = AlphaMLP(input_size=16, hidden_sizes=(16, 4), output_name=output_name)
    assert net.get_output_names() == (output_name,)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_alpha_mlp_forward_2d(
    device: str, batch_size: int, input_size: int, output_size: int, mode: bool
):
    device = torch.device(device)
    mlp = AlphaMLP(input_size=input_size, hidden_sizes=(16, output_size)).to(device=device)
    mlp.train(mode)
    out = mlp(torch.randn(batch_size, input_size, device=device))
    assert out.shape == (batch_size, output_size)
    assert out.device == device
    assert out.dtype == torch.float


def test_alpha_mlp_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=AlphaMLP(input_size=6, hidden_sizes=(8, 4)),
            criterion=VanillaLoss(criterion=nn.MSELoss()),
        ),
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
    )


#############################
#     Tests for BetaMLP     #
#############################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_beta_mlp_forward_2d(
    device: str, batch_size: int, input_size: int, output_size: int, mode: bool
):
    device = torch.device(device)
    mlp = BetaMLP(input_size=input_size, hidden_sizes=(16, output_size)).to(device=device)
    mlp.train(mode)
    out = mlp(torch.randn(batch_size, input_size, device=device))
    assert out.shape == (batch_size, output_size)
    assert out.device == device
    assert out.dtype == torch.float


def test_beta_mlp_is_loss_decreasing() -> None:
    assert is_loss_decreasing_with_sgd(
        model=VanillaModel(
            network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
            criterion=VanillaLoss(criterion=nn.MSELoss()),
        ),
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
    )


######################################
#     Tests for create_alpha_mlp     #
######################################


@mark.parametrize("input_size", SIZES)
def test_create_alpha_mlp_input_size(input_size: int):
    assert (
        create_alpha_mlp(input_size=input_size, hidden_sizes=(16, 4)).linear1.in_features
        == input_size
    )


@mark.parametrize("hidden_sizes,num_layers", (((8,), 2), ((16, 4), 4), ((16, 16, 4), 6)))
def test_create_alpha_mlp_num_layers(hidden_sizes: Sequence[int], num_layers: int):
    assert len(create_alpha_mlp(input_size=16, hidden_sizes=hidden_sizes)) == num_layers


@mark.parametrize("hidden_sizes,num_layers", (((8,), 3), ((16, 4), 6), ((16, 16, 4), 9)))
def test_create_alpha_mlp_num_layers_with_dropout(hidden_sizes: Sequence[int], num_layers: int):
    assert (
        len(create_alpha_mlp(input_size=16, hidden_sizes=hidden_sizes, dropout=0.5)) == num_layers
    )


@mark.parametrize("hidden_size", SIZES)
def test_create_alpha_mlp_hidden_size(hidden_size: int):
    mlp = create_alpha_mlp(input_size=16, hidden_sizes=(hidden_size, 8))
    assert mlp.linear1.out_features == hidden_size
    assert mlp.linear2.in_features == hidden_size


@mark.parametrize("output_size", SIZES)
def test_create_alpha_mlp_output_size(output_size: int):
    assert (
        create_alpha_mlp(input_size=16, hidden_sizes=(32, output_size)).linear2.out_features
        == output_size
    )


def test_create_alpha_mlp_activation_relu() -> None:
    mlp = create_alpha_mlp(input_size=16, hidden_sizes=(16, 4))
    assert isinstance(mlp.relu1, nn.ReLU)
    assert isinstance(mlp.relu2, nn.ReLU)


def test_create_alpha_mlp_activation_gelu() -> None:
    mlp = create_alpha_mlp(
        input_size=16,
        hidden_sizes=(16, 4),
        activation={OBJECT_TARGET: "torch.nn.ELU", "alpha": 0.1},
    )
    assert isinstance(mlp.elu1, nn.ELU)
    assert isinstance(mlp.elu2, nn.ELU)


@mark.parametrize("dropout", DROPOUT_VALUES)
def test_create_alpha_mlp_dropout(dropout: float) -> None:
    mlp = create_alpha_mlp(input_size=16, hidden_sizes=(16, 4), dropout=dropout)
    assert mlp.dropout1.p == dropout
    assert mlp.dropout2.p == dropout


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_create_alpha_mlp_forward_2d(
    device: str, batch_size: int, input_size: int, output_size: int
):
    device = torch.device(device)
    mlp = create_alpha_mlp(input_size=input_size, hidden_sizes=(4, output_size)).to(device=device)
    out = mlp(torch.randn(batch_size, input_size, device=device))
    assert out.device == device
    assert out.dtype == torch.float
    assert out.shape == (batch_size, output_size)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_create_alpha_mlp_forward_3d(
    device: str, seq_len: int, batch_size: int, input_size: int, output_size: int
):
    device = torch.device(device)
    mlp = create_alpha_mlp(input_size=input_size, hidden_sizes=(4, output_size)).to(device=device)
    out = mlp(torch.randn(seq_len, batch_size, input_size, device=device))
    assert out.device == device
    assert out.dtype == torch.float
    assert out.shape == (seq_len, batch_size, output_size)


#####################################
#     Tests for create_beta_mlp     #
#####################################


@mark.parametrize("input_size", SIZES)
def test_create_beta_mlp_input_size(input_size: int):
    assert (
        create_beta_mlp(input_size=input_size, hidden_sizes=(16, 4)).linear1.in_features
        == input_size
    )


@mark.parametrize("hidden_sizes,num_layers", (((8,), 1), ((16, 4), 3), ((16, 16, 4), 5)))
def test_create_beta_mlp_num_layers(hidden_sizes: Sequence[int], num_layers: int):
    assert len(create_beta_mlp(input_size=16, hidden_sizes=hidden_sizes)) == num_layers


@mark.parametrize("hidden_sizes,num_layers", (((8,), 2), ((16, 4), 5), ((16, 16, 4), 8)))
def test_create_beta_mlp_num_layers_with_dropout(hidden_sizes: Sequence[int], num_layers: int):
    assert len(create_beta_mlp(input_size=16, hidden_sizes=hidden_sizes, dropout=0.5)) == num_layers


@mark.parametrize("hidden_size", SIZES)
def test_create_beta_mlp_hidden_size(hidden_size: int):
    mlp = create_beta_mlp(input_size=16, hidden_sizes=(hidden_size, 8))
    assert mlp.linear1.out_features == hidden_size
    assert mlp.linear2.in_features == hidden_size


@mark.parametrize("output_size", SIZES)
def test_create_beta_mlp_output_size(output_size: int):
    assert (
        create_beta_mlp(input_size=16, hidden_sizes=(32, output_size)).linear2.out_features
        == output_size
    )


def test_create_beta_mlp_activation_relu() -> None:
    assert isinstance(create_beta_mlp(input_size=16, hidden_sizes=(16, 4)).relu1, nn.ReLU)


def test_create_beta_mlp_activation_elu() -> None:
    assert isinstance(
        create_beta_mlp(
            input_size=16,
            hidden_sizes=(16, 4),
            activation={OBJECT_TARGET: "torch.nn.ELU", "alpha": 0.1},
        ).elu1,
        nn.ELU,
    )


@mark.parametrize("dropout", DROPOUT_VALUES)
def test_create_beta_mlp_dropout(dropout: float) -> None:
    mlp = create_beta_mlp(input_size=16, hidden_sizes=(16, 4), dropout=dropout)
    assert mlp.dropout1.p == dropout
    assert mlp.dropout2.p == dropout


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_create_beta_mlp_forward_2d(
    device: str, batch_size: int, input_size: int, output_size: int
):
    device = torch.device(device)
    mlp = create_beta_mlp(input_size=input_size, hidden_sizes=(4, output_size)).to(device=device)
    out = mlp(torch.randn(batch_size, input_size, device=device))
    assert out.device == device
    assert out.dtype == torch.float
    assert out.shape == (batch_size, output_size)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_create_beta_mlp_forward_3d(
    device: str, seq_len: int, batch_size: int, input_size: int, output_size: int
):
    device = torch.device(device)
    mlp = create_beta_mlp(input_size=input_size, hidden_sizes=(4, output_size)).to(device=device)
    out = mlp(torch.randn(seq_len, batch_size, input_size, device=device))
    assert out.device == device
    assert out.dtype == torch.float
    assert out.shape == (seq_len, batch_size, output_size)
