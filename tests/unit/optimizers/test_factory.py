from objectory import OBJECT_TARGET
from torch import nn
from torch.optim import SGD

from gravitorch.optimizers import setup_optimizer

#####################################
#     Tests for setup_optimizer     #
#####################################


def test_setup_optimizer_none() -> None:
    assert setup_optimizer(model=nn.Linear(4, 6), optimizer=None) is None


def test_setup_optimizer_object() -> None:
    model = nn.Linear(4, 6)
    optimizer = SGD(params=model.parameters(), lr=0.01)
    assert setup_optimizer(model=model, optimizer=optimizer) is optimizer


def test_setup_optimizer_dict_no_learnable_parameters() -> None:
    assert (
        setup_optimizer(
            model=nn.Identity(), optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        )
        is None
    )


def test_setup_optimizer_dict_learnable_parameters() -> None:
    assert isinstance(
        setup_optimizer(
            model=nn.Linear(4, 6), optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        ),
        SGD,
    )
