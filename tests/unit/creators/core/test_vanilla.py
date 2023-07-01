from unittest.mock import Mock

from objectory import OBJECT_TARGET
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from gravitorch import constants as ct
from gravitorch.creators.core import VanillaCoreCreator
from gravitorch.datasources import BaseDataSource
from gravitorch.testing import DummyDataSource

########################################
#     Tests for VanillaCoreCreator     #
########################################


def test_vanilla_core_creator_str() -> None:
    assert str(VanillaCoreCreator(DummyDataSource(), nn.Linear(4, 6))).startswith(
        "VanillaCoreCreator("
    )


def test_vanilla_core_creator_create() -> None:
    engine = Mock()
    creator = VanillaCoreCreator(
        datasource=DummyDataSource(),
        model=nn.Linear(4, 6),
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, BaseDataSource)
    assert isinstance(model, nn.Module)
    assert optimizer is None
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 2
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)


def test_vanilla_core_creator_create_optimizer() -> None:
    engine = Mock()
    creator = VanillaCoreCreator(
        datasource=DummyDataSource(),
        model=nn.Linear(4, 6),
        optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, BaseDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 3
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)


def test_vanilla_core_creator_create_lr_scheduler() -> None:
    engine = Mock()
    creator = VanillaCoreCreator(
        datasource=DummyDataSource(),
        model=nn.Linear(4, 6),
        optimizer={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        lr_scheduler={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, BaseDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert isinstance(lr_scheduler, StepLR)
    assert len(engine.add_module.call_args_list) == 4
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)
    assert engine.add_module.call_args_list[3].args == (ct.LR_SCHEDULER, lr_scheduler)
