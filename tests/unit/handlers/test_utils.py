from __future__ import annotations

import logging
from unittest.mock import Mock

from minevent import EventHandler
from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, mark
from torch.nn import Identity

from gravitorch.engines import BaseEngine
from gravitorch.handlers import (
    EpochLRMonitor,
    add_unique_event_handler,
    is_handler_config,
    setup_and_attach_handlers,
    setup_handler,
    to_events,
)

EVENTS = ("my_event", "my_other_event")


##############################################
#     Tests for add_unique_event_handler     #
##############################################


@mark.parametrize("event", EVENTS)
def test_add_unique_event_handler_has_event_handler_false(event: str) -> None:
    engine = Mock(spec=BaseEngine)
    engine.has_event_handler.return_value = False
    event_handler = EventHandler(Mock())
    add_unique_event_handler(engine, event, event_handler)
    engine.add_event_handler.assert_called_once_with(event, event_handler)


@mark.parametrize("event", EVENTS)
def test_add_unique_event_handler_has_event_handler_true(event: str) -> None:
    engine = Mock(spec=BaseEngine)
    engine.has_event_handler.return_value = True
    event_handler = EventHandler(Mock())
    add_unique_event_handler(engine, event, event_handler)
    engine.add_event_handler.assert_not_called()


#######################################
#     Tests for is_handler_config     #
#######################################


def test_is_handler_config_true() -> None:
    assert is_handler_config({OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"})


def test_is_handler_config_false() -> None:
    assert not is_handler_config({OBJECT_TARGET: "torch.nn.Identity"})


###################################
#     Tests for setup_handler     #
###################################


def test_setup_handler_from_object() -> None:
    handler = setup_handler(handler=EpochLRMonitor())
    assert isinstance(handler, EpochLRMonitor)


def test_setup_handler_from_config() -> None:
    assert isinstance(
        setup_handler(handler={OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"}), EpochLRMonitor
    )


def test_setup_handler_incorrect_type(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_handler({OBJECT_TARGET: "torch.nn.Identity"}), Identity)
        assert caplog.messages


###############################################
#     Tests for setup_and_attach_handlers     #
###############################################


def test_setup_and_attach_handlers_from_config() -> None:
    engine = Mock(spec=BaseEngine)
    engine.epoch = -1
    engine.has_event_handler.return_value = False
    handlers = setup_and_attach_handlers(
        engine=engine, handlers=[{OBJECT_TARGET: "gravitorch.handlers.EpochLRMonitor"}]
    )
    assert len(handlers) == 1
    assert isinstance(handlers[0], EpochLRMonitor)


def test_setup_and_attach_handlers_2_handlers() -> None:
    engine = Mock(spec=BaseEngine)
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
