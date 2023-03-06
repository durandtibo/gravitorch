import logging

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from gravitorch.lr_schedulers import setup_lr_scheduler

########################################
#     Tests for setup_lr_scheduler     #
########################################


def test_setup_lr_scheduler_optimizer_none() -> None:
    lr_scheduler = StepLR(optimizer=SGD(nn.Linear(4, 6).parameters(), lr=0.01), step_size=5)
    assert setup_lr_scheduler(optimizer=None, lr_scheduler=lr_scheduler) is None


def test_setup_lr_scheduler_lr_scheduler_none() -> None:
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    assert setup_lr_scheduler(optimizer=optimizer, lr_scheduler=None) is None


def test_setup_lr_scheduler_lr_scheduler_object_correct_optimizer(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
        lr_scheduler = StepLR(optimizer=optimizer, step_size=5)
        assert setup_lr_scheduler(optimizer=optimizer, lr_scheduler=lr_scheduler) is lr_scheduler
        assert len(caplog.messages) == 0


def test_setup_lr_scheduler_lr_scheduler_object_incorrect_optimizer(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        optimizer1 = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
        optimizer2 = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
        lr_scheduler = StepLR(optimizer=optimizer2, step_size=5)
        assert setup_lr_scheduler(optimizer=optimizer1, lr_scheduler=lr_scheduler) is lr_scheduler
        assert len(caplog.messages) == 1


def test_setup_lr_scheduler_lr_scheduler_dict() -> None:
    optimizer = SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    lr_scheduler = setup_lr_scheduler(
        optimizer=optimizer,
        lr_scheduler={OBJECT_TARGET: "torch.optim.lr_scheduler.StepLR", "step_size": 5},
    )
    assert isinstance(lr_scheduler, StepLR)
    assert lr_scheduler.optimizer == optimizer
