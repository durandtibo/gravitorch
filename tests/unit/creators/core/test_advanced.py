from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import fixture
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from gravitorch import constants as ct
from gravitorch.creators.core import AdvancedCoreCreator
from gravitorch.creators.datasource import BaseDataSourceCreator, DataSourceCreator
from gravitorch.creators.lr_scheduler import VanillaLRSchedulerCreator
from gravitorch.creators.model import BaseModelCreator, ModelCreator
from gravitorch.creators.optimizer import OptimizerCreator
from gravitorch.engines import BaseEngine
from gravitorch.testing import DummyDataSource

#########################################
#     Tests for AdvancedCoreCreator     #
#########################################


@fixture
def datasource_creator() -> BaseDataSourceCreator:
    return DataSourceCreator(config={OBJECT_TARGET: "gravitorch.testing.DummyDataSource"})


@fixture
def model_creator() -> BaseModelCreator:
    return ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
    )


def test_advanced_core_creator_str(
    datasource_creator: BaseDataSourceCreator, model_creator: BaseModelCreator
) -> None:
    assert str(
        AdvancedCoreCreator(datasource_creator=datasource_creator, model_creator=model_creator)
    ).startswith("AdvancedCoreCreator(")


def test_advanced_core_creator_create_no_optimizer_creator(
    datasource_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
) -> None:
    engine = Mock(spec=BaseEngine)
    creator = AdvancedCoreCreator(
        datasource_creator=datasource_creator, model_creator=model_creator
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, DummyDataSource)
    assert isinstance(model, nn.Module)
    assert optimizer is None
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 2
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)


def test_advanced_core_creator_create_optimizer_creator(
    datasource_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
) -> None:
    engine = Mock(spec=BaseEngine)
    creator = AdvancedCoreCreator(
        datasource_creator=datasource_creator,
        model_creator=model_creator,
        optimizer_creator=OptimizerCreator(
            optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        ),
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, DummyDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert lr_scheduler is None
    assert len(engine.add_module.call_args_list) == 3
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)


def test_advanced_core_creator_create_lr_scheduler_creator(
    datasource_creator: BaseDataSourceCreator,
    model_creator: BaseModelCreator,
) -> None:
    engine = Mock(spec=BaseEngine)
    creator = AdvancedCoreCreator(
        datasource_creator=datasource_creator,
        model_creator=model_creator,
        optimizer_creator=OptimizerCreator(
            optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        ),
        lr_scheduler_creator=VanillaLRSchedulerCreator(
            lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        ),
    )
    datasource, model, optimizer, lr_scheduler = creator.create(engine=engine)
    assert isinstance(datasource, DummyDataSource)
    assert isinstance(model, nn.Module)
    assert isinstance(optimizer, SGD)
    assert isinstance(lr_scheduler, StepLR)
    assert len(engine.add_module.call_args_list) == 4
    assert engine.add_module.call_args_list[0].args == (ct.DATA_SOURCE, datasource)
    assert engine.add_module.call_args_list[1].args == (ct.MODEL, model)
    assert engine.add_module.call_args_list[2].args == (ct.OPTIMIZER, optimizer)
    assert engine.add_module.call_args_list[3].args == (ct.LR_SCHEDULER, lr_scheduler)
