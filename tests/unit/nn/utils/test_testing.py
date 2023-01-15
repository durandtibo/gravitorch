import torch
from torch import nn
from torch.optim import SGD

from gravitorch.nn.utils import (
    is_loss_decreasing,
    is_loss_decreasing_with_adam,
    is_loss_decreasing_with_sgd,
)

########################################
#     Tests for is_loss_decreasing     #
########################################


def test_is_loss_decreasing_true():
    module = nn.Linear(4, 2)
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


def test_is_loss_decreasing_false():
    module = nn.Linear(4, 2)
    assert not is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
        num_iterations=0,
    )


def test_is_loss_decreasing_train_mode():
    module = nn.Linear(4, 2)
    module.train()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
    assert module.training


def test_is_loss_decreasing_eval_mode():
    module = nn.Linear(4, 2)
    module.eval()
    assert is_loss_decreasing(
        module=module,
        criterion=nn.MSELoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
    assert not module.training


def test_is_loss_decreasing_criterion_functional():
    module = nn.Linear(4, 2)
    optimizer = SGD(module.parameters(), lr=0.01)
    assert is_loss_decreasing(
        module=module,
        criterion=nn.functional.mse_loss,
        optimizer=optimizer,
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


##################################################
#     Tests for is_loss_decreasing_with_adam     #
##################################################


def test_is_loss_decreasing_with_adam_true():
    assert is_loss_decreasing_with_adam(
        module=nn.Linear(4, 2),
        criterion=nn.MSELoss(),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )


#################################################
#     Tests for is_loss_decreasing_with_sgd     #
#################################################


def test_is_loss_decreasing_with_sgd_true():
    assert is_loss_decreasing_with_sgd(
        module=nn.Linear(4, 2),
        criterion=nn.MSELoss(),
        feature=torch.rand(4, 4),
        target=torch.rand(4, 2),
    )
