import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark, raises

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import EpochSysInfoMonitor
from gravitorch.utils.events import ConditionalEventHandler, EpochPeriodicCondition

EVENTS = ("my_event", "my_other_event")


#########################################
#     Tests for EpochSysInfoMonitor     #
#########################################


def test_epoch_sysinfo_monitor_str():
    assert str(EpochSysInfoMonitor()).startswith("EpochSysInfoMonitor(")


@mark.parametrize("event", EVENTS)
def test_epoch_sysinfo_monitor_event(event: str):
    assert EpochSysInfoMonitor(event)._event == event


def test_epoch_sysinfo_monitor_event_default():
    assert EpochSysInfoMonitor()._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_epoch_sysinfo_monitor_freq(freq: int):
    assert EpochSysInfoMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_sysinfo_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        EpochSysInfoMonitor(freq=freq)


def test_epoch_sysinfo_monitor_freq_default():
    assert EpochSysInfoMonitor()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_sysinfo_monitor_attach(event: str, freq: int):
    handler = EpochSysInfoMonitor(event=event, freq=freq)
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


def test_epoch_sysinfo_monitor_attach_duplicate():
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochSysInfoMonitor().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_epoch_sysinfo_monitor_monitor(caplog: LogCaptureFixture):
    engine = Mock(spec=BaseEngine, epoch=4)
    handler = EpochSysInfoMonitor()
    with caplog.at_level(logging.INFO):
        handler.monitor(engine)
    assert len(caplog.messages) == 3
