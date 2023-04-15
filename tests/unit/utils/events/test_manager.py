import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from pytest import fixture, mark, raises

from gravitorch.events import VanillaEventHandler
from gravitorch.events.manager import EventManager, to_event_handlers_str

logger = logging.getLogger(__name__)

EVENTS = ("my_event", "my_other_event")


def trace(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        wrapper.called = True
        wrapper.call_count += 1
        wrapper.args = args
        wrapper.kwargs = kwargs
        return func(*args, **kwargs)

    wrapper.called = False
    wrapper.call_count = 0
    wrapper.args = ()
    wrapper.kwargs = {}
    return wrapper


@trace
def hello_handler() -> None:
    r"""Implements a simple handler that prints hello."""
    logger.info("Hello!")


@fixture(scope="function", autouse=True)
def reset_tracer() -> None:
    def reset_func(func: Callable) -> None:
        func.called = False
        func.call_count = 0
        func.args = ()
        func.kwargs = {}

    reset_func(hello_handler)


##################################
#     Tests for EventManager     #
##################################


def test_event_manager_str_without_event_handler() -> None:
    assert str(EventManager()).startswith("EventManager(")


def test_event_manager_str_with_event_handler() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert str(event_manager).startswith("EventManager(")


def test_event_manager_last_fired_event_none() -> None:
    assert EventManager().last_fired_event is None


def test_last_fired_event_name_after_fire() -> None:
    event_manager = EventManager()
    event_manager.fire_event("my_event")
    assert event_manager.last_fired_event == "my_event"
    event_manager.fire_event("my_other_event")
    assert event_manager.last_fired_event == "my_other_event"
    event_manager.fire_event("my_event")
    assert event_manager.last_fired_event == "my_event"


def test_event_manager_last_fired_event() -> None:
    event_manager = EventManager()
    event_manager.fire_event("my_event")
    event_manager.fire_event("my_other_event")
    assert event_manager.last_fired_event == "my_other_event"


@mark.parametrize("event", EVENTS)
def test_event_manager_add_event_handler_event(event: str) -> None:
    event_manager = EventManager()
    event_manager.add_event_handler(event, VanillaEventHandler(hello_handler))
    assert len(event_manager._event_handlers[event]) == 1


def test_event_manager_add_event_handler_duplicate_event_handlers() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert len(event_manager._event_handlers["my_event"]) == 2
    assert (
        event_manager._event_handlers["my_event"][0] == event_manager._event_handlers["my_event"][1]
    )


@mark.parametrize("event", EVENTS)
def test_event_manager_fire_event(event: str) -> None:
    event_manager = EventManager()
    event_manager.add_event_handler(event, VanillaEventHandler(hello_handler))
    event_manager.fire_event(event)
    assert hello_handler.called
    assert hello_handler.call_count == 1


def test_event_manager_fire_event_2_times() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.fire_event("my_event")
    event_manager.fire_event("my_event")
    assert hello_handler.called
    assert hello_handler.call_count == 2


def test_event_manager_fire_event_without_event_handler() -> None:
    event_manager = EventManager()
    event_manager.fire_event("my_event")
    assert event_manager.last_fired_event == "my_event"


def test_event_manager_has_event_handler_true() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert event_manager.has_event_handler(VanillaEventHandler(hello_handler))


def test_event_manager_has_event_handler_true_with_event() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert event_manager.has_event_handler(VanillaEventHandler(hello_handler), event="my_event")


def test_event_manager_has_event_handler_false() -> None:
    event_manager = EventManager()
    assert not event_manager.has_event_handler(VanillaEventHandler(hello_handler))


def test_event_manager_has_event_handler_false_with_event() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(lambda *args, **kwargs: True))
    event_manager.add_event_handler("my_other_event", VanillaEventHandler(hello_handler))
    assert not event_manager.has_event_handler(VanillaEventHandler(hello_handler), event="my_event")


@mark.parametrize("event", EVENTS)
def test_event_manager_remove_event_handler(event: str) -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.remove_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert len(event_manager._event_handlers["my_event"]) == 0


def test_event_manager_remove_event_handler_duplicate_handler() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.add_event_handler("my_event", VanillaEventHandler(lambda *args, **kwargs: True))
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.remove_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert len(event_manager._event_handlers["my_event"]) == 1


def test_event_manager_remove_event_handler_multiple_events() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.add_event_handler("my_other_event", VanillaEventHandler(hello_handler))
    event_manager.remove_event_handler("my_event", VanillaEventHandler(hello_handler))
    assert len(event_manager._event_handlers["my_event"]) == 0
    assert len(event_manager._event_handlers["my_other_event"]) == 1


def test_event_manager_remove_event_handler_missing_event() -> None:
    event_manager = EventManager()
    with raises(ValueError, match="'my_event' event does not exist"):
        event_manager.remove_event_handler("my_event", VanillaEventHandler(hello_handler))


def test_event_manager_remove_event_handler_missing_handler() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(lambda *args, **kwargs: True))
    with raises(
        ValueError, match="is not found among registered event handlers for 'my_event' event"
    ):
        event_manager.remove_event_handler("my_event", VanillaEventHandler(hello_handler))


def test_event_manager_reset() -> None:
    event_manager = EventManager()
    event_manager.add_event_handler("my_event", VanillaEventHandler(hello_handler))
    event_manager.add_event_handler("my_other_event", VanillaEventHandler(hello_handler))
    event_manager.fire_event("my_event")
    event_manager.reset()
    assert len(event_manager._event_handlers) == 0
    assert event_manager.last_fired_event is None


def test_event_manager_reset_empty() -> None:
    event_manager = EventManager()
    assert len(event_manager._event_handlers) == 0
    assert event_manager.last_fired_event is None


def test_event_manager_no_duplicate() -> None:
    event_manager = EventManager()
    for _ in range(5):
        event_handler = VanillaEventHandler(hello_handler)
        if not event_manager.has_event_handler(event_handler, "my_event"):
            event_manager.add_event_handler("my_event", event_handler)
    assert len(event_manager._event_handlers["my_event"]) == 1


###########################################
#     Tests for to_event_handlers_str     #
###########################################


def test_to_event_handlers_str_empty() -> None:
    assert to_event_handlers_str({}) == ""


def test_to_event_handlers_str() -> None:
    assert to_event_handlers_str(
        {"event1": ["handler1"], "event2": ["handler21", "handler22\nblabla"]}
    ) == ("(event1)\n  (0) handler1\n(event2)\n  (0) handler21\n  (1) handler22\n    blabla")


def test_to_event_handlers_str_num_spaces_4() -> None:
    print(
        to_event_handlers_str(
            {"event1": ["handler1"], "event2": ["handler21", "handler22\nblabla"]},
            num_spaces=4,
        )
    )
    assert (
        to_event_handlers_str(
            {"event1": ["handler1"], "event2": ["handler21", "handler22\nblabla"]},
            num_spaces=4,
        )
        == "(event1)\n    (0) handler1\n(event2)\n    (0) handler21\n    (1) handler22\n        blabla"
    )
