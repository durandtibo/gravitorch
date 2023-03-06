from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.observers import (
    BaseLoopObserver,
    NoOpLoopObserver,
    SequentialLoopObserver,
)

############################################
#     Tests for SequentialLoopObserver     #
############################################


def test_sequential_loop_observer_str() -> None:
    assert str(
        SequentialLoopObserver((NoOpLoopObserver(), Mock(spec=BaseLoopObserver)))
    ).startswith("SequentialLoopObserver(")


def test_sequential_loop_observer_start() -> None:
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    SequentialLoopObserver((observer1, observer2)).start(engine)
    observer1.start.assert_called_once_with(engine)
    observer2.start.assert_called_once_with(engine)


def test_sequential_loop_observer_end() -> None:
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    SequentialLoopObserver((observer1, observer2)).end(engine)
    observer1.end.assert_called_once_with(engine)
    observer2.end.assert_called_once_with(engine)


def test_sequential_loop_observer_update() -> None:
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    SequentialLoopObserver((observer1, observer2)).update(engine, {}, 1)
    observer1.update.assert_called_once_with(engine, {}, 1)
    observer2.update.assert_called_once_with(engine, {}, 1)
