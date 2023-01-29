from unittest.mock import Mock, patch

from torch.backends import cuda

from gravitorch.engines import BaseEngine
from gravitorch.experimental.rsrc import PyTorchCudaBackend
from gravitorch.experimental.runners import BaseResourceRunner, TrainingRunner
from gravitorch.handlers import BaseHandler
from gravitorch.utils.exp_trackers import BaseExpTracker

########################################
#     Tests for BaseResourceRunner     #
########################################


class FakeResourceRunner(BaseResourceRunner):
    r"""Defines a fake runner to test ``BaseResourceRunner`` because it has an
    abstract method."""

    def _run(self) -> int:
        return 42


def test_base_resource_runner_run_without_resources():
    assert FakeResourceRunner().run() == 42


def test_base_resource_runner_run_with_resources():
    default = cuda.matmul.allow_tf32
    assert FakeResourceRunner(resources=(PyTorchCudaBackend(allow_tf32=True),)).run() == 42
    assert cuda.matmul.allow_tf32 == default


####################################
#     Tests for TrainingRunner     #
####################################


def test_training_runner_str():
    assert str(TrainingRunner(engine={})).startswith("TrainingRunner(")


def test_training_runner_run_no_handler():
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    runner = TrainingRunner(engine=engine, exp_tracker=exp_tracker, random_seed=42)
    with patch("gravitorch.experimental.runners.resource._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=tuple(),
        random_seed=42,
    )


def test_training_runner_run_handler():
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    handlers = tuple([Mock(spec=BaseHandler)])
    runner = TrainingRunner(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )
    with patch("gravitorch.experimental.runners.resource._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )


def test_training_runner_run_no_exp_tracker():
    engine = Mock(spec=BaseEngine)
    runner = TrainingRunner(engine=engine, random_seed=42)
    with patch("gravitorch.experimental.runners.resource._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=None,
        handlers=tuple(),
        random_seed=42,
    )


def test_training_runner_engine():
    engine = Mock()
    TrainingRunner(engine).run()
    engine.train.assert_called_once()
