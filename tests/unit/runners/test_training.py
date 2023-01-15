from unittest.mock import Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch.runners.training import TrainingRunner, _run_training_pipeline
from gravitorch.utils.exp_trackers import NoOpExpTracker

DIST_BACKENDS = ("auto", "gloo", "nccl", None)


####################################
#     Tests for TrainingRunner     #
####################################


def test_training_runner_str():
    assert str(TrainingRunner(engine={})).startswith("TrainingRunner(")


def test_training_runner_engine():
    engine = Mock()
    TrainingRunner(engine, dist_backend=None).run()
    engine.train.assert_called_once()


@mark.parametrize("random_seed", (1, 2, 3))
def test_training_runner_random_seed(random_seed: int):
    engine = Mock()
    TrainingRunner(engine, random_seed=random_seed).run()
    x1 = torch.rand(2, 3)
    torch.manual_seed(random_seed)
    x2 = torch.rand(2, 3)
    assert x1.equal(x2)


def test_training_runner_run_train():
    engine = Mock()
    TrainingRunner(engine, tuple(), NoOpExpTracker()).run()
    engine.train.assert_called_once()


def test_training_runner_run_setup_and_attach_handlers():
    engine = Mock()
    handlers = Mock()
    runner = TrainingRunner(engine, handlers, exp_tracker=NoOpExpTracker())
    with patch("gravitorch.runners.training.setup_and_attach_handlers") as setup_mock:
        runner.run()
        setup_mock.assert_called_once_with(engine, handlers)


############################################
#     Tests for _run_training_pipeline     #
############################################


def test_run_training_pipeline_engine_config():
    with patch("gravitorch.runners.training.BaseEngine.factory") as engine_factory:
        exp_tracker = NoOpExpTracker()
        _run_training_pipeline(
            {OBJECT_TARGET: "MyEngine", "max_epochs": 15}, handlers=tuple(), exp_tracker=exp_tracker
        )
        engine_factory.assert_called_once_with(
            _target_="MyEngine", max_epochs=15, exp_tracker=exp_tracker
        )


def test_run_training_pipeline_setup_and_attach_handlers():
    engine = Mock()
    handlers = Mock()
    with patch("gravitorch.runners.training.setup_and_attach_handlers") as setup_mock:
        _run_training_pipeline(engine, handlers, exp_tracker=None)
        setup_mock.assert_called_once_with(engine, handlers)


def test_run_training_pipeline_exp_tracker_none():
    engine = Mock()
    _run_training_pipeline(engine, handlers=tuple(), exp_tracker=None)
    engine.train.assert_called_once()


def test_run_training_pipeline_exp_tracker_config():
    engine = Mock()
    _run_training_pipeline(
        engine,
        handlers=tuple(),
        exp_tracker={OBJECT_TARGET: "gravitorch.utils.exp_trackers.NoOpExpTracker"},
    )
    engine.train.assert_called_once()


@patch("gravitorch.runners.training.dist.is_distributed", lambda *args, **kwargs: True)
def test_run_training_pipeline_distributed():
    engine = Mock()
    _run_training_pipeline(engine, handlers=tuple(), exp_tracker=None)
    engine.train.assert_called_once()
