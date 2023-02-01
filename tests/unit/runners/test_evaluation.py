from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET

from gravitorch.engines import BaseEngine
from gravitorch.handlers import BaseHandler
from gravitorch.runners.evaluation import EvaluationRunner, _run_evaluation_pipeline
from gravitorch.utils.exp_trackers import BaseExpTracker, NoOpExpTracker

DIST_BACKENDS = ("auto", "gloo", "nccl", None)


######################################
#     Tests for EvaluationRunner     #
######################################


def test_evaluation_runner_str():
    assert str(EvaluationRunner(engine={})).startswith("EvaluationRunner(")


def test_evaluation_runner_run_no_handler():
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    runner = EvaluationRunner(engine=engine, exp_tracker=exp_tracker, random_seed=42)
    with patch("gravitorch.runners.evaluation._run_evaluation_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=tuple(),
        random_seed=42,
    )


def test_evaluation_runner_run_handler():
    engine = Mock(spec=BaseEngine)
    exp_tracker = Mock(spec=BaseExpTracker)
    handlers = tuple([Mock(spec=BaseHandler)])
    runner = EvaluationRunner(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )
    with patch("gravitorch.runners.evaluation._run_evaluation_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=exp_tracker,
        handlers=handlers,
        random_seed=42,
    )


def test_evaluation_runner_run_no_exp_tracker():
    engine = Mock(spec=BaseEngine)
    runner = EvaluationRunner(engine=engine, random_seed=42)
    with patch("gravitorch.runners.evaluation._run_evaluation_pipeline") as pipe_mock:
        runner.run()
    pipe_mock.assert_called_once_with(
        engine=engine,
        exp_tracker=None,
        handlers=tuple(),
        random_seed=42,
    )


def test_evaluation_runner_engine():
    engine = Mock()
    EvaluationRunner(engine).run()
    engine.eval.assert_called_once()


##############################################
#     Tests for _run_evaluation_pipeline     #
##############################################


def test_run_evaluation_pipeline_engine_config():
    with patch("gravitorch.runners.evaluation.BaseEngine.factory") as engine_factory:
        exp_tracker = NoOpExpTracker()
        _run_evaluation_pipeline(
            {OBJECT_TARGET: "MyEngine", "max_epochs": 15},
            handlers=tuple(),
            exp_tracker=exp_tracker,
        )
        engine_factory.assert_called_once_with(
            _target_="MyEngine", max_epochs=15, exp_tracker=exp_tracker
        )


def test_run_evaluation_pipeline_setup_and_attach_handlers():
    engine = Mock()
    handlers = Mock()
    with patch("gravitorch.runners.evaluation.setup_and_attach_handlers") as setup_mock:
        _run_evaluation_pipeline(engine, handlers, exp_tracker=None)
        setup_mock.assert_called_once_with(engine, handlers)


def test_run_evaluation_pipeline_exp_tracker_none():
    engine = Mock()
    _run_evaluation_pipeline(engine, handlers=tuple(), exp_tracker=None)
    engine.eval.assert_called_once()


def test_run_evaluation_pipeline_exp_tracker_config():
    engine = Mock()
    _run_evaluation_pipeline(
        engine,
        handlers=tuple(),
        exp_tracker={OBJECT_TARGET: "gravitorch.utils.exp_trackers.NoOpExpTracker"},
    )
    engine.eval.assert_called_once()


@patch("gravitorch.runners.evaluation.dist.is_distributed", lambda *args, **kwargs: True)
def test_run_evaluation_pipeline_distributed():
    engine = Mock()
    _run_evaluation_pipeline(engine, handlers=tuple(), exp_tracker=None)
    engine.eval.assert_called_once()
