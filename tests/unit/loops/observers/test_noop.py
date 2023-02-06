from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.observers import NoOpLoopObserver

######################################
#     Tests for NoOpLoopObserver     #
######################################


def test_noop_observer_str():
    assert str(NoOpLoopObserver()).startswith("NoOpLoopObserver()")


def test_noop_observer_start():
    NoOpLoopObserver().start(engine=Mock(spec=BaseEngine))


def test_noop_observer_end():
    NoOpLoopObserver().end(engine=Mock(spec=BaseEngine))


def test_noop_observer_update():
    NoOpLoopObserver().update(engine=Mock(spec=BaseEngine), model_input={}, model_output={})
