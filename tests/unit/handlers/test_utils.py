from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch.handlers import (
    EpochLRMonitor,
    add_unique_event_handler,
    setup_and_attach_handlers,
    setup_handler,
    to_events,
)
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


##############################################
#     Tests for add_unique_event_handler     #
##############################################


@mark.parametrize("event", EVENTS)
def test_add_unique_event_handler_has_event_handler_false(event: str) -> None:
    engine = Mock()
    engine.has_event_handler.return_value = False
    event_handler = VanillaEventHandler(Mock())
    add_unique_event_handler(engine, event, event_handler)
    engine.add_event_handler.assert_called_once_with(event, event_handler)


@mark.parametrize("event", EVENTS)
def test_add_unique_event_handler_has_event_handler_true(event: str) -> None:
    engine = Mock()
    engine.has_event_handler.return_value = True
    event_handler = VanillaEventHandler(Mock())
    add_unique_event_handler(engine, event, event_handler)
    engine.add_event_handler.assert_not_called()


###################################
#     Tests for setup_handler     #
###################################


def test_setup_handler_from_object() -> None:
    handler = setup_handler(handler=EpochLRMonitor())
    assert isinstance(handler, EpochLRMonitor)


def test_setup_handler_from_config() -> None:
    handler = setup_handler(handler={OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"})
    assert isinstance(handler, EpochLRMonitor)


###############################################
#     Tests for setup_and_attach_handlers     #
###############################################


def test_setup_and_attach_handlers_from_config() -> None:
    engine = Mock()
    engine.epoch = -1
    engine.has_event_handler.return_value = False
    handlers = setup_and_attach_handlers(
        engine=engine, handlers=[{OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"}]
    )
    assert len(handlers) == 1
    assert isinstance(handlers[0], EpochLRMonitor)


def test_setup_and_attach_handlers_2_handlers() -> None:
    engine = Mock()
    handler1 = Mock()
    handler2 = Mock()
    handlers = setup_and_attach_handlers(engine=engine, handlers=(handler1, handler2))
    assert len(handlers) == 2
    handler1.attach.assert_called_once_with(engine)
    handler2.attach.assert_called_once_with(engine)


###############################
#     Tests for to_events     #
###############################


def test_to_events_str() -> None:
    assert to_events("my_event") == ("my_event",)


def test_to_events_list() -> None:
    assert to_events(["my_event", "my_other_event"]) == ("my_event", "my_other_event")


def test_to_events_tuple() -> None:
    assert to_events(("my_event", "my_other_event")) == ("my_event", "my_other_event")
