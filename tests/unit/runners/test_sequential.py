from unittest.mock import Mock

from gravitorch.runners import BaseRunner, SequentialRunner

######################################
#     Tests for SequentialRunner     #
######################################


def test_sequential_runner_str() -> None:
    assert str(SequentialRunner([]))


def test_sequential_runner_run_empty() -> None:
    assert SequentialRunner([]).run() is None


def test_sequential_runner_run() -> None:
    runner1 = Mock(spec=BaseRunner)
    runner2 = Mock(spec=BaseRunner)
    assert SequentialRunner([runner1, runner2]).run() is None
    runner1.run.assert_called_once_with()
    runner2.run.assert_called_once_with()
