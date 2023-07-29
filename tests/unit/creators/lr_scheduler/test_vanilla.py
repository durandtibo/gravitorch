import logging
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

from gravitorch import constants as ct
from gravitorch.creators.lr_scheduler import VanillaLRSchedulerCreator
from gravitorch.engines import BaseEngine

###############################################
#     Tests for VanillaLRSchedulerCreator     #
###############################################


def test_vanilla_lr_scheduler_creator_str() -> None:
    assert str(VanillaLRSchedulerCreator()).startswith("VanillaLRSchedulerCreator(")


@mark.parametrize("add_module_to_engine", (True, False))
def test_vanilla_lr_scheduler_creator_add_module_to_engine(add_module_to_engine: bool) -> None:
    assert (
        VanillaLRSchedulerCreator(add_module_to_engine=add_module_to_engine)._add_module_to_engine
        == add_module_to_engine
    )


def test_vanilla_lr_scheduler_creator_create_lr_scheduler_config_none() -> None:
    creator = VanillaLRSchedulerCreator()
    assert creator.create(engine=Mock(spec=BaseEngine), optimizer=Mock(spec=Optimizer)) is None


@mark.parametrize("step_size", (1, 5))
def test_vanilla_lr_scheduler_creator_create_lr_scheduler_config_dict(step_size: int) -> None:
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={
            OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR",
            "step_size": step_size,
        },
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=Mock(spec=BaseEngine), optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    assert lr_scheduler.step_size == step_size


def test_vanilla_lr_scheduler_creator_create_add_module_to_engine_true() -> None:
    engine = Mock(spec=BaseEngine)
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    engine.add_module.assert_called_once_with(ct.LR_SCHEDULER, lr_scheduler)


def test_vanilla_lr_scheduler_creator_create_add_module_to_engine_false() -> None:
    engine = Mock(spec=BaseEngine)
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        add_module_to_engine=False,
    )
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
    assert isinstance(lr_scheduler, StepLR)
    engine.add_module.assert_not_called()


def test_vanilla_lr_scheduler_creator_create_lr_scheduler_handler() -> None:
    engine = Mock(spec=BaseEngine)
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
        engine = Mock(spec=BaseEngine)
        creator = VanillaLRSchedulerCreator(
            lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
        )
        optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
        lr_scheduler = creator.create(engine=engine, optimizer=optimizer)
        assert isinstance(lr_scheduler, StepLR)
        assert caplog.messages  # The user should see a warning


def test_vanilla_lr_scheduler_creator_create_optimizer_none() -> None:
    creator = VanillaLRSchedulerCreator(
        lr_scheduler_config={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    assert creator.create(engine=Mock(spec=BaseEngine), optimizer=None) is None
