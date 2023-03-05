import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from gravitorch import constants as ct
from gravitorch.creators.lr_scheduler import VanillaLRSchedulerCreator

###############################################
#     Tests for VanillaLRSchedulerCreator     #
###############################################


def test_vanilla_lr_scheduler_creator_str():
    assert str(VanillaLRSchedulerCreator()).startswith("VanillaLRSchedulerCreator(")


@mark.parametrize("add_module_to_engine", (True, False))
def test_vanilla_lr_scheduler_creator_add_module_to_engine(add_module_to_engine: bool):
    assert (
        VanillaLRSchedulerCreator(add_module_to_engine=add_module_to_engine)._add_module_to_engine
        == add_module_to_engine
    )


def test_vanilla_lr_scheduler_creator_create_lr_scheduler_config_none():
    creator = VanillaLRSchedulerCreator()
    assert creator.create(engine=Mock(), optimizer=Mock()) is None


@mark.parametrize("step_size", (1, 5))
def test_vanilla_lr_scheduler_creator_create_lr_scheduler_config_dict(step_size: int):
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={
            OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR",
            "step_size": step_size,
        },
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=Mock(), optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    assert lr_scheduler.step_size == step_size


def test_vanilla_lr_scheduler_creator_create_add_module_to_engine_true():
    engine = Mock()
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    engine.add_module.assert_called_once_with(ct.LR_SCHEDULER, lr_scheduler)


def test_vanilla_lr_scheduler_creator_create_add_module_to_engine_false():
    engine = Mock()
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        add_module_to_engine=False,
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    engine.add_module.assert_not_called()


def test_vanilla_lr_scheduler_creator_create_lr_scheduler_handler():
    engine = Mock()
    lr_scheduler_handler = Mock()
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        lr_scheduler_handler=lr_scheduler_handler,
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    lr_scheduler_handler.attach.assert_called_once_with(engine=engine)


def test_vanilla_lr_scheduler_creator_create_no_lr_scheduler_handler(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARN):
        engine = Mock()
        creator = VanillaLRSchedulerCreator(
            lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        )
        optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
        lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
        assert isinstance(lr_scheduler, StepLR)
        assert len(caplog.messages) == 1  # The user should see a warning


def test_vanilla_lr_scheduler_creator_create_optimizer_none():
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    assert creator.create(engine=Mock(), optimizer=None) is None
