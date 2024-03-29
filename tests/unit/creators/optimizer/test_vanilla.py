from unittest.mock import Mock

import torch.optim
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.optimizer import OptimizerCreator
from gravitorch.engines import BaseEngine
from gravitorch.optimizers.utils import get_learning_rate_per_group

######################################
#     Tests for OptimizerCreator     #
######################################


def test_optimizer_creator_str() -> None:
    assert str(OptimizerCreator()).startswith("OptimizerCreator(")


@mark.parametrize("add_module_to_engine", (True, False))
def test_optimizer_creator_add_module_to_engine(add_module_to_engine: bool) -> None:
    assert (
        OptimizerCreator(add_module_to_engine=add_module_to_engine)._add_module_to_engine
        == add_module_to_engine
    )


def test_optimizer_creator_create_optimizer_config_none() -> None:
    creator = OptimizerCreator()
    assert creator.create(engine=Mock(spec=BaseEngine), model=Mock()) is None


@mark.parametrize("lr", (0.01, 0.001))
def test_optimizer_creator_create_optimizer_config_dict(lr: float) -> None:
    creator = OptimizerCreator(optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": lr})
    optimizer = creator.create(engine=Mock(spec=BaseEngine), model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    assert get_learning_rate_per_group(optimizer) == {0: lr}


def test_optimizer_creator_create_optimizer_add_module_to_engine_true() -> None:
    engine = Mock(spec=BaseEngine)
    creator = OptimizerCreator(optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01})
    optimizer = creator.create(engine=engine, model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    engine.add_module.assert_called_once_with(ct.OPTIMIZER, optimizer)


def test_optimizer_creator_create_optimizer_add_module_to_engine_false() -> None:
    engine = Mock(spec=BaseEngine)
    creator = OptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        add_module_to_engine=False,
    )
    optimizer = creator.create(engine=engine, model=nn.Linear(4, 6))
    assert isinstance(optimizer, torch.optim.SGD)
    engine.add_module.assert_not_called()
