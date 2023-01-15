from unittest.mock import Mock

from gravitorch.handlers import (
    EpochLRMonitor,
    EpochLRSchedulerHandler,
    EpochLRSchedulerUpdater,
    IterationLRMonitor,
    IterationLRSchedulerHandler,
    IterationLRSchedulerUpdater,
    VanillaLRSchedulerHandler,
)

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


###############################################
#     Tests for VanillaLRSchedulerHandler     #
###############################################


def test_vanilla_lr_scheduler_handler_str():
    assert str(
        VanillaLRSchedulerHandler(lr_scheduler_updater=Mock(), lr_monitor=Mock())
    ).startswith("VanillaLRSchedulerHandler(")


def test_vanilla_lr_scheduler_handler_attach():
    lr_scheduler_updater = Mock()
    lr_monitor = Mock()
    handler = VanillaLRSchedulerHandler(
        lr_scheduler_updater=lr_scheduler_updater, lr_monitor=lr_monitor
    )
    engine = Mock()
    handler.attach(engine)
    lr_scheduler_updater.attach.assert_called_once_with(engine)
    lr_monitor.attach.assert_called_once_with(engine)


#############################################
#     Tests for EpochLRSchedulerHandler     #
#############################################


def test_epoch_lr_scheduler_handler():
    handler = EpochLRSchedulerHandler()
    assert isinstance(handler._lr_scheduler_updater, EpochLRSchedulerUpdater)
    assert isinstance(handler._lr_monitor, EpochLRMonitor)


#################################################
#     Tests for IterationLRSchedulerHandler     #
#################################################


def test_iteration_lr_scheduler_handler():
    handler = IterationLRSchedulerHandler()
    assert isinstance(handler._lr_scheduler_updater, IterationLRSchedulerUpdater)
    assert isinstance(handler._lr_monitor, IterationLRMonitor)
