from pathlib import Path
from unittest.mock import Mock

from pytest import raises

from gravitorch.runners import NoRepeatRunner
from gravitorch.utils.io import save_text

####################################
#     Tests for NoRepeatRunner     #
####################################


def test_no_repeat_runner_str(tmp_path: Path):
    assert str(NoRepeatRunner(path=tmp_path, runner=Mock())).startswith("NoRepeatRunner(")


def test_no_repeat_runner_run_successful(tmp_path: Path):
    internal_runner = Mock(run=Mock())
    runner = NoRepeatRunner(path=tmp_path, runner=internal_runner)
    runner.run()
    internal_runner.run.assert_called_once()
    assert runner._success_path.is_file()


def test_no_repeat_runner_run_already_successful(tmp_path: Path):
    internal_runner = Mock(run=Mock())
    runner = NoRepeatRunner(path=tmp_path, runner=internal_runner)
    save_text("something", runner._success_path)
    assert runner._success_path.is_file()
    runner.run()
    internal_runner.run.assert_not_called()


def test_no_repeat_runner_run_not_successful(tmp_path: Path):
    internal_runner = Mock(run=Mock(side_effect=RuntimeError("Test")))
    runner = NoRepeatRunner(path=tmp_path, runner=internal_runner)
    with raises(Exception):
        runner.run()
    assert not runner._success_path.is_file()
