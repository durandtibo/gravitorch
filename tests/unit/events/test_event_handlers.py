import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from minevent import PeriodicCondition
from pytest import fixture, raises

from gravitorch.events import ConditionalEventHandler, VanillaEventHandler

logger = logging.getLogger(__name__)


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
    print("Hello!")


@trace
def hello_name_handler(first_name: str, last_name: str) -> None:
    r"""Implements a simple handler that prints hello and the name of
    the person."""
    logger.info(f"Hello. I am {first_name} {last_name}")


@fixture(scope="function", autouse=True)
def reset_tracer() -> None:
    def reset_func(func: Callable) -> None:
        func.called = False
        func.call_count = 0
        func.args = ()
        func.kwargs = {}

    reset_func(hello_handler)
    reset_func(hello_name_handler)


#########################################
#     Tests for VanillaEventHandler     #
#########################################


def test_vanilla_event_handler_str() -> None:
    assert str(VanillaEventHandler(hello_handler)).startswith("VanillaEventHandler(")


def test_vanilla_event_handler_eq_true() -> None:
    assert VanillaEventHandler(hello_handler) == VanillaEventHandler(hello_handler)


def test_vanilla_event_handler_eq_false_same_class() -> None:
    assert VanillaEventHandler(hello_handler) != VanillaEventHandler(
        hello_handler, handler_args=("something",)
    )


def test_vanilla_event_handler_eq_false_different_classes() -> None:
    assert VanillaEventHandler(hello_handler) != ConditionalEventHandler(
        hello_name_handler, PeriodicCondition(3)
    )


def test_vanilla_event_handler_without_args_and_kwargs() -> None:
    event_handler = VanillaEventHandler(hello_handler)
    assert event_handler.handler == hello_handler
    assert event_handler.handler_args == ()
    assert event_handler.handler_kwargs == {}


def test_vanilla_event_handler_with_only_args() -> None:
    event_handler = VanillaEventHandler(hello_name_handler, handler_args=("John", "Doe"))
    assert event_handler.handler == hello_name_handler
    assert event_handler.handler_args == ("John", "Doe")
    assert event_handler.handler_kwargs == {}


def test_vanilla_event_handler_with_only_kwargs() -> None:
    event_handler = VanillaEventHandler(
        hello_name_handler, handler_kwargs={"first_name": "John", "last_name": "Doe"}
    )
    assert event_handler.handler == hello_name_handler
    assert event_handler.handler_args == ()
    assert event_handler.handler_kwargs == {"first_name": "John", "last_name": "Doe"}


def test_vanilla_event_handler_with_args_and_kwargs() -> None:
    event_handler = VanillaEventHandler(
        hello_handler, handler_args=("John",), handler_kwargs={"last_name": "Doe"}
    )
    assert event_handler.handler == hello_handler
    assert event_handler.handler_args == ("John",)
    assert event_handler.handler_kwargs == {"last_name": "Doe"}


def test_vanilla_event_handler_handle_without_args_and_kwargs() -> None:
    VanillaEventHandler(hello_handler).handle()
    assert hello_handler.called
    assert hello_handler.call_count == 1
    assert hello_handler.args == ()
    assert hello_handler.kwargs == {}


def test_vanilla_event_handler_handle_with_only_args() -> None:
    VanillaEventHandler(hello_name_handler, handler_args=("John", "Doe")).handle()
    assert hello_name_handler.called
    assert hello_name_handler.call_count == 1
    assert hello_name_handler.args == ("John", "Doe")
    assert hello_name_handler.kwargs == {}


def test_vanilla_event_handler_handle_with_only_kwargs() -> None:
    VanillaEventHandler(
        hello_name_handler, handler_kwargs={"first_name": "John", "last_name": "Doe"}
    ).handle()
    assert hello_name_handler.called
    assert hello_name_handler.call_count == 1
    assert hello_name_handler.args == ()
    assert hello_name_handler.kwargs == {"first_name": "John", "last_name": "Doe"}


def test_vanilla_event_handler_handle_with_args_and_kwargs() -> None:
    VanillaEventHandler(
        hello_name_handler, handler_args=("John",), handler_kwargs={"last_name": "Doe"}
    ).handle()
    assert hello_name_handler.called
    assert hello_name_handler.call_count == 1
    assert hello_name_handler.args == ("John",)
    assert hello_name_handler.kwargs == {"last_name": "Doe"}


def test_vanilla_event_handler_handle_called_2_times() -> None:
    event_handler = VanillaEventHandler(hello_handler)
    event_handler.handle()
    event_handler.handle()
    assert hello_handler.call_count == 2


#############################################
#     Tests for ConditionalEventHandler     #
#############################################


def test_conditional_event_handler_str() -> None:
    assert str(ConditionalEventHandler(hello_handler, PeriodicCondition(2))).startswith(
        "ConditionalEventHandler("
    )


def test_conditional_event_handler_eq_true() -> None:
    assert ConditionalEventHandler(hello_handler, PeriodicCondition(3)) == ConditionalEventHandler(
        hello_handler, PeriodicCondition(3)
    )


def test_conditional_event_handler_eq_false_same_class() -> None:
    assert ConditionalEventHandler(hello_handler, PeriodicCondition(3)) != ConditionalEventHandler(
        hello_handler, PeriodicCondition(2)
    )


def test_conditional_event_handler_eq_false_different_classes() -> None:
    assert ConditionalEventHandler(hello_handler, PeriodicCondition(3)) != VanillaEventHandler(
        hello_name_handler
    )


def test_conditional_event_handler_callable_condition() -> None:
    event_handler = ConditionalEventHandler(hello_handler, PeriodicCondition(3))
    assert event_handler.handler == hello_handler
    assert event_handler.handler_args == ()
    assert event_handler.handler_kwargs == {}
    assert event_handler.condition == PeriodicCondition(3)


def test_conditional_event_handler_non_callable_condition() -> None:
    with raises(TypeError, match="The condition is not callable"):
        ConditionalEventHandler(hello_handler, 123)


def test_vanilla_event_handler_handle_1() -> None:
    ConditionalEventHandler(hello_handler, PeriodicCondition(3)).handle()
    assert hello_handler.called
    assert hello_handler.call_count == 1
    assert hello_handler.args == ()
    assert hello_handler.kwargs == {}


def test_vanilla_event_handler_handle_10() -> None:
    event_handler = ConditionalEventHandler(hello_handler, PeriodicCondition(3))
    for _ in range(10):
        event_handler.handle()
    assert hello_handler.called
    assert hello_handler.call_count == 4
    assert hello_handler.args == ()
    assert hello_handler.kwargs == {}
