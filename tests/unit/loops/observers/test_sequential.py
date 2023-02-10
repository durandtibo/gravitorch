from unittest.mock import Mock

from gravitorch.engines import BaseEngine
from gravitorch.loops.observers import BaseLoopObserver, NoOpLoopObserver, Sequential

################################
#     Tests for Sequential     #
################################


def test_sequential_str():
    assert str(Sequential((NoOpLoopObserver(), Mock(spec=BaseLoopObserver)))).startswith(
        "Sequential("
    )


def test_sequential_start():
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    Sequential((observer1, observer2)).start(engine)
    observer1.start.assert_called_once_with(engine)
    observer2.start.assert_called_once_with(engine)


def test_sequential_end():
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    Sequential((observer1, observer2)).end(engine)
    observer1.end.assert_called_once_with(engine)
    observer2.end.assert_called_once_with(engine)


def test_sequential_update():
    engine = Mock(spec=BaseEngine)
    observer1 = Mock(spec=BaseLoopObserver)
    observer2 = Mock(spec=BaseLoopObserver)
    Sequential((observer1, observer2)).update(engine, {}, 1)
    observer1.update.assert_called_once_with(engine, {}, 1)
    observer2.update.assert_called_once_with(engine, {}, 1)
