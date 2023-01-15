from unittest.mock import Mock

from pytest import mark, raises
from torch import nn
from torch.optim import SGD

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import EpochLRMonitor, IterationLRMonitor
from gravitorch.utils.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

EVENTS = ("my_event", "my_other_event")


####################################
#     Tests for EpochLRMonitor     #
####################################


def test_epoch_lr_monitor_str():
    assert str(EpochLRMonitor()).startswith("EpochLRMonitor(")


@mark.parametrize("event", EVENTS)
def test_epoch_lr_monitor_event(event: str):
    assert EpochLRMonitor(event)._event == event


def test_epoch_lr_monitor_event_default():
    assert EpochLRMonitor()._event == EngineEvents.TRAIN_EPOCH_STARTED


@mark.parametrize("freq", (1, 2))
def test_epoch_lr_monitor_freq(freq: int):
    assert EpochLRMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_lr_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        EpochLRMonitor(freq=freq)


def test_epoch_lr_monitor_freq_default():
    assert EpochLRMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_lr_monitor_attach(event: str, freq: int):
    handler = EpochLRMonitor(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=EpochPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_epoch_lr_monitor_attach_duplicate():
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochLRMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_epoch_lr_monitor_monitor():
    engine = Mock(spec=BaseEngine, epoch=1, optimizer=SGD(nn.Linear(4, 6).parameters(), lr=0.01))
    EpochLRMonitor().monitor(engine)
    engine.log_metrics.assert_called_once_with(
        {"epoch/optimizer.group0.lr": 0.01}, step=EpochStep(1)
    )


def test_epoch_lr_monitor_monitor_no_optimizer():
    engine = Mock(spec=BaseEngine, epoch=1, optimizer=None)
    EpochLRMonitor().monitor(engine)
    engine.log_metrics.assert_not_called()


########################################
#     Tests for IterationLRMonitor     #
########################################


def test_iteration_lr_monitor_str():
    assert str(IterationLRMonitor()).startswith("IterationLRMonitor(")


@mark.parametrize("event", EVENTS)
def test_iteration_lr_monitor_event(event: str):
    assert IterationLRMonitor(event)._event == event


def test_iteration_lr_monitor_event_default():
    assert IterationLRMonitor()._event == EngineEvents.TRAIN_ITERATION_STARTED


@mark.parametrize("freq", (1, 2))
def test_iteration_lr_monitor_freq(freq: int):
    assert IterationLRMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_iteration_lr_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        IterationLRMonitor(freq=freq)


def test_iteration_lr_monitor_freq_default():
    assert IterationLRMonitor()._freq == 10


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_iteration_lr_monitor_attach(event: str, freq: int):
    handler = IterationLRMonitor(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=IterationPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_iteration_lr_monitor_attach_duplicate():
    engine = Mock(spec=BaseEngine, iteration=-1, has_event_handler=Mock(return_value=True))
    IterationLRMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_iteration_lr_monitor_monitor():
    engine = Mock(
        spec=BaseEngine, iteration=10, optimizer=SGD(nn.Linear(4, 6).parameters(), lr=0.01)
    )
    IterationLRMonitor().monitor(engine)
    engine.log_metrics.assert_called_once_with(
        {"iteration/optimizer.group0.lr": 0.01}, step=IterationStep(10)
    )


def test_iteration_lr_monitor_monitor_no_optimizer():
    engine = Mock(spec=BaseEngine, iteration=10, optimizer=None)
    IterationLRMonitor().monitor(engine)
    engine.log_metrics.assert_not_called()
