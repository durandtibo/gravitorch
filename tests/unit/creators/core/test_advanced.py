from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import fixture
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from gravitorch import constants as ct
from gravitorch.creators.core import AdvancedCoreCreator
from gravitorch.creators.datasource import (
    BaseDataSourceCreator,
    VanillaDataSourceCreator,
)
from gravitorch.creators.lr_scheduler import VanillaLRSchedulerCreator
from gravitorch.creators.model import BaseModelCreator, VanillaModelCreator
from gravitorch.creators.optimizer import VanillaOptimizerCreator
from tests.unit.engines.util import FakeDataSource

#########################################
#     Tests for AdvancedCoreCreator     #
#########################################


@fixture
def data_source_creator() -> BaseDataSourceCreator:
    return VanillaDataSourceCreator(
        config={OBJECT_TARGET: "tests.unit.engines.util.FakeDataSource"}
    )


@fixture
def model_creator() -> BaseModelCreator:
    return VanillaModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
    )


def test_advanced_core_creator_str(
    data_source_creator: BaseDataSourceCreator, model_creator: BaseModelCreator
):
    assert str(
        AdvancedCoreCreator(data_source_creator=data_source_creator, model_creator=model_creator)
    ).startswith("AdvancedCoreCreator(")


def test_advanced_core_creator_create_no_optimizer_creator(
    data_source_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
):
    engine = Mock()
    creator = AdvancedCoreCreator(
        data_source_creator=data_source_creator, model_creator=model_creator
    )
    data_source, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(data_source, FakeDataSource)
    assert isinstance(model, nn.Module)
    assert optimizer is None
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 2
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, data_source)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)


def test_advanced_core_creator_create_optimizer_creator(
    data_source_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
):
    engine = Mock()
    creator = AdvancedCoreCreator(
        data_source_creator=data_source_creator,
        model_creator=model_creator,
        optimizer_creator=VanillaOptimizerCreator(
            optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        ),
    )
    data_source, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(data_source, FakeDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 3
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, data_source)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)


def test_advanced_core_creator_create_lr_scheduler_creator(
    data_source_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
):
    engine = Mock()
    creator = AdvancedCoreCreator(
        data_source_creator=data_source_creator,
        model_creator=model_creator,
        optimizer_creator=VanillaOptimizerCreator(
            optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        ),
        lr_scheduler_creator=VanillaLRSchedulerCreator(
            lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ),
    )
    data_source, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(data_source, FakeDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert isinstance(lr_scheduler, StepLR)
    assert len(engine.add_module.call_args_list) == 4
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, data_source)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)
    assert engine.add_module.call_args_list[3].args == (ct.LR_SCHEDULER, lr_scheduler)
