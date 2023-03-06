from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.observers import NoOpLoopObserver

######################################
#     Tests for NoOpLoopObserver     #
######################################


def test_noop_observer_str() -> None:
    assert str(NoOpLoopObserver()).startswith("NoOpLoopObserver()")


def test_noop_observer_start() -> None:
    NoOpLoopObserver().start(engine=Mock(spec=BaseEngine))


def test_noop_observer_end() -> None:
    NoOpLoopObserver().end(engine=Mock(spec=BaseEngine))


def test_noop_observer_update() -> None:
    NoOpLoopObserver().update(engine=Mock(spec=BaseEngine), model_input={}, model_output={})
