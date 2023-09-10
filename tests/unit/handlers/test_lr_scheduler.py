from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.handlers import (
    EpochLRMonitor,
    EpochLRScheduler,
    EpochLRSchedulerUpdater,
    IterationLRMonitor,
    IterationLRScheduler,
    IterationLRSchedulerUpdater,
    LRScheduler,
)

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


#################################
#     Tests for LRScheduler     #
#################################


def test_vanilla_lr_scheduler_str() -> None:
    assert str(LRScheduler(lr_scheduler_updater=Mock(), lr_monitor=Mock())).startswith(
        "LRScheduler("
    )


def test_vanilla_lr_scheduler_attach() -> None:
    lr_scheduler_updater = Mock()
    lr_monitor = Mock()
    handler = LRScheduler(lr_scheduler_updater=lr_scheduler_updater, lr_monitor=lr_monitor)
    engine = Mock(spec=BaseEngine)
    handler.attach(engine)
    lr_scheduler_updater.attach.assert_called_once_with(engine)
    lr_monitor.attach.assert_called_once_with(engine)


######################################
#     Tests for EpochLRScheduler     #
######################################


def test_epoch_lr_scheduler() -> None:
    handler = EpochLRScheduler()
    assert isinstance(handler._lr_scheduler_updater, EpochLRSchedulerUpdater)
    assert isinstance(handler._lr_monitor, EpochLRMonitor)


##########################################
#     Tests for IterationLRScheduler     #
##########################################


def test_iteration_lr_scheduler() -> None:
    handler = IterationLRScheduler()
    assert isinstance(handler._lr_scheduler_updater, IterationLRSchedulerUpdater)
    assert isinstance(handler._lr_monitor, IterationLRMonitor)
