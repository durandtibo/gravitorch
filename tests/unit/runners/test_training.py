from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.engines import BaseEngine
from gravitorch.handlers import BaseHandler
from gravitorch.runners.training import TrainingRunner, _run_training_pipeline
from gravitorch.utils.exp_trackers import BaseExpTracker, NoOpExpTracker

DIST_BACKENDS = ("auto", "gloo", "nccl", None)


####################################
#     Tests for TrainingRunner     #
####################################


def test_training_runner_str() -> None:
    assert str(TrainingRunner(engine={})).startswith("TrainingRunner(")


def test_training_runner_run_no_handler() -> None:
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    runner = TrainingRunner(engine=engine, exp_tracker=exp_tracker, random_seed=42)
    with patch("gravitorch.runners.training._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=(),
        random_seed=42,
    )


def test_training_runner_run_handler() -> None:
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    handlers = (Mock(spec=BaseHandler),)
    runner = TrainingRunner(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )
    with patch("gravitorch.runners.training._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )


def test_training_runner_run_no_exp_tracker() -> None:
    engine = Mock(spec=BaseEngine)
    runner = TrainingRunner(engine=engine, random_seed=42)
    with patch("gravitorch.runners.training._run_training_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=None,
        handlers=(),
        random_seed=42,
    )


def test_training_runner_engine() -> None:
    engine = Mock()
    TrainingRunner(engine).run()
    engine.train.assert_called_once()


############################################
#     Tests for _run_training_pipeline     #
############################################


def test_run_training_pipeline_engine_config() -> None:
    with patch("gravitorch.runners.training.BaseEngine.factory") as engine_factory:
        exp_tracker = NoOpExpTracker()
        _run_training_pipeline(
            {OBJECT_TARGET: "MyEngine", "max_epochs": 15}, handlers=(), exp_tracker=exp_tracker
        )
        engine_factory.assert_called_once_with(
            _target_="MyEngine", max_epochs=15, exp_tracker=exp_tracker
        )


def test_run_training_pipeline_setup_and_attach_handlers() -> None:
    engine = Mock()
    handlers = Mock()
    with patch("gravitorch.runners.training.setup_and_attach_handlers") as setup_mock:
        _run_training_pipeline(engine, handlers, exp_tracker=None)
        setup_mock.assert_called_once_with(engine, handlers)


def test_run_training_pipeline_exp_tracker_none() -> None:
    engine = Mock()
    _run_training_pipeline(engine, handlers=(), exp_tracker=None)
    engine.train.assert_called_once()


def test_run_training_pipeline_exp_tracker_config() -> None:
    engine = Mock()
    _run_training_pipeline(
        engine,
        handlers=(),
        exp_tracker={OBJECT_TARGET: "gravitorch.utils.exp_trackers.NoOpExpTracker"},
    )
    engine.train.assert_called_once()


@patch("gravitorch.runners.training.dist.is_distributed", lambda *args, **kwargs: True)
def test_run_training_pipeline_distributed() -> None:
    engine = Mock()
    _run_training_pipeline(engine, handlers=(), exp_tracker=None)
    engine.train.assert_called_once()
