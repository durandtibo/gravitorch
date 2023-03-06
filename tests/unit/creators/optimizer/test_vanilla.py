from unittest.mock import Mock

import torch.optim
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.optimizer import VanillaOptimizerCreator
from gravitorch.optimizers.utils import get_learning_rate_per_group

#############################################
#     Tests for VanillaOptimizerCreator     #
#############################################


def test_vanilla_optimizer_creator_str():
    assert str(VanillaOptimizerCreator()).startswith("VanillaOptimizerCreator(")


@mark.parametrize("add_module_to_engine", (True, False))
def test_vanilla_optimizer_creator_add_module_to_engine(add_module_to_engine: bool) -> None:
    assert (
        VanillaOptimizerCreator(add_module_to_engine=add_module_to_engine)._add_module_to_engine
        == add_module_to_engine
    )


def test_vanilla_optimizer_creator_create_optimizer_config_none():
    creator = VanillaOptimizerCreator()
    assert creator.create(engine=Mock(), model=Mock()) is None


@mark.parametrize("lr", (0.01, 0.001))
def test_vanilla_optimizer_creator_create_optimizer_config_dict(lr: float) -> None:
    creator = VanillaOptimizerCreator(optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": lr})
    optimizer = creator.create(engine=Mock(), model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    assert get_learning_rate_per_group(optimizer) == {0: lr}


def test_vanilla_optimizer_creator_create_optimizer_add_module_to_engine_true():
    engine = Mock()
    creator = VanillaOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
    )
    optimizer = creator.create(engine=engine, model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    engine.add_module.assert_called_once_with(ct.OPTIMIZER, optimizer)


def test_vanilla_optimizer_creator_create_optimizer_add_module_to_engine_false():
    engine = Mock()
    creator = VanillaOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        add_module_to_engine=False,
    )
    optimizer = creator.create(engine=engine, model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    engine.add_module.assert_not_called()
