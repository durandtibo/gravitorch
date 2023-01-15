from unittest.mock import Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch.runners.evaluation import EvaluationRunner, _run_evaluation_pipeline
from gravitorch.utils.exp_trackers import NoOpExpTracker

DIST_BACKENDS = ("auto", "gloo", "nccl", None)


######################################
#     Tests for EvaluationRunner     #
######################################


def test_evaluation_runner_str():
    assert str(EvaluationRunner(engine={})).startswith("EvaluationRunner(")


def test_evaluation_runner_engine():
    engine = Mock()
    EvaluationRunner(engine, dist_backend=None).run()
    engine.eval.assert_called_once()


@mark.parametrize("random_seed", (1, 2, 3))
def test_evaluation_runner_random_seed(random_seed: int):
    engine = Mock()
    EvaluationRunner(engine, random_seed=random_seed).run()
    x1 = torch.rand(2, 3)
    torch.manual_seed(random_seed)
    x2 = torch.rand(2, 3)
    assert x1.equal(x2)


def test_evaluation_runner_run_train():
    engine = Mock()
    EvaluationRunner(engine, tuple(), NoOpExpTracker()).run()
    engine.eval.assert_called_once()


def test_evaluation_runner_run_setup_and_attach_handlers():
    engine = Mock()
    handlers = Mock()
    runner = EvaluationRunner(engine, handlers, exp_tracker=NoOpExpTracker())
    with patch("gravitorch.runners.evaluation.setup_and_attach_handlers") as setup_mock:
        runner.run()
        setup_mock.assert_called_once_with(engine, handlers)


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
