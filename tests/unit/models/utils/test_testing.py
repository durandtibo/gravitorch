import torch
from torch import nn
from torch.optim import SGD

from gravitorch import constants as ct
from gravitorch.models import VanillaModel
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.networks import BetaMLP
from gravitorch.models.utils import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)

########################################
#     Tests for is_loss_decreasing     #
########################################


def test_is_loss_decreasing_true_train_mode() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    assert is_loss_decreasing(
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
    )


def test_is_loss_decreasing_false() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    assert not is_loss_decreasing(
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
        num_iterations=0,
    )


def test_is_loss_decreasing_train_mode() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    model.train()
    optimizer = SGD(model.parameters(), lr=0.01)
    assert is_loss_decreasing(
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
    )
    assert model.training


def test_is_loss_decreasing_eval_mode() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    model.eval()
    optimizer = SGD(model.parameters(), lr=0.01)
    assert is_loss_decreasing(
        model=model,
        optimizer=optimizer,
        batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)},
    )
    assert not model.training


def test_is_loss_decreasing_with_adam_true() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    assert is_loss_decreasing_with_adam(
        model=model, batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)}
    )


def test_is_loss_decreasing_with_sgd_true() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=6, hidden_sizes=(8, 4)),
        criterion=VanillaLoss(criterion=nn.MSELoss()),
    )
    assert is_loss_decreasing_with_sgd(
        model=model, batch={ct.INPUT: torch.randn(2, 6), ct.TARGET: torch.randn(2, 4)}
    )
