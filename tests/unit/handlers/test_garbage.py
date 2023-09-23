from unittest.mock import Mock

from pytest import mark, raises

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.engines.events import EpochPeriodicCondition
from gravitorch.handlers import EpochGarbageCollector
from gravitorch.utils.events import GConditionalEventHandler

EVENTS = ("my_event", "my_other_event")


############################################
#     Tests for EpochGarbageCollection     #
############################################


def test_epoch_garbage_collector_str() -> None:
    assert str(EpochGarbageCollector()).startswith("EpochGarbageCollector(")


@mark.parametrize("event", EVENTS)
def test_epoch_garbage_collector_event(event: str) -> None:
    assert EpochGarbageCollector(event)._event == event


def test_epoch_garbage_collector_event_default() -> None:
    assert EpochGarbageCollector()._event == EngineEvents.EPOCH_COMPLETED


@mark.parametrize("freq", (1, 2))
def test_epoch_garbage_collector_freq(freq: int) -> None:
    assert EpochGarbageCollector(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_garbage_collector_incorrect_freq(freq: int) -> None:
    with raises(ValueError, match="freq has to be greater than 0"):
        EpochGarbageCollector(freq=freq)


def test_epoch_garbage_collector_freq_default() -> None:
    assert EpochGarbageCollector()._freq == 1


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_garbage_collector_attach(event: str, freq: int) -> None:
    handler = EpochGarbageCollector(event=event, freq=freq)
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        GConditionalEventHandler(
            handler.collect,
            condition=EpochPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_epoch_garbage_collector_attach_duplicate() -> None:
    engine = Mock(spec=BaseEngine, epoch=-1, has_event_handler=Mock(return_value=True))
    EpochGarbageCollector().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_epoch_garbage_collector_collect() -> None:
    engine = Mock(spec=BaseEngine)
    EpochGarbageCollector().collect(engine)
