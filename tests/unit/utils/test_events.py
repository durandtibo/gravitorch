from __future__ import annotations

from minevent import PeriodicCondition

from gravitorch.utils.events import GConditionalEventHandler, GEventHandler


def hello_handler() -> None:
    r"""Implements a simple handler that prints hello."""
    print("Hello!")


###################################
#     Tests for GEventHandler     #
###################################


def test_gevent_handler_str() -> None:
    assert repr(GEventHandler(hello_handler)).startswith("GEventHandler(")


def test_gevent_handler_repr() -> None:
    assert str(GEventHandler(hello_handler)).startswith("GEventHandler(")


##############################################
#     Tests for GConditionalEventHandler     #
##############################################


def test_gconditional_event_handler_str() -> None:
    assert repr(GConditionalEventHandler(hello_handler, PeriodicCondition(2))).startswith(
        "GConditionalEventHandler("
    )


def test_conditional_event_handler_repr() -> None:
    assert str(GConditionalEventHandler(hello_handler, PeriodicCondition(2))).startswith(
        "GConditionalEventHandler("
    )
