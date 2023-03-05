from pathlib import Path
from typing import Any, Union
from unittest.mock import Mock

import torch
from pytest import TempPathFactory, fixture, mark, raises
from torch import nn

from gravitorch.utils.artifacts import JSONArtifact
from gravitorch.utils.exp_trackers import (
    EpochStep,
    IterationStep,
    NoOpExpTracker,
    NotActivatedExpTrackerError,
)
from gravitorch.utils.integrations import is_matplotlib_available, is_pillow_available

if is_matplotlib_available():
    from matplotlib.pyplot import Figure
else:
    Figure = "matplotlib.pyplot.Figure"  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image
else:
    Image = "PIL.Image.Image"  # pragma: no cover

####################################
#     Tests for NoOpExpTracker     #
####################################

# The following tests check that the NoOp implementation does not generate errors.


@fixture(scope="module")
def tracker(tmp_path_factory: TempPathFactory) -> NoOpExpTracker:
    with NoOpExpTracker(experiment_path=tmp_path_factory.mktemp("data").as_posix()) as tracker:
        yield tracker


def test_noop_exp_tracker_str(tracker: NoOpExpTracker) -> None:
    assert str(tracker).startswith("NoOpExpTracker(")


def test_noop_exp_tracker_artifact_path(tracker: NoOpExpTracker) -> None:
    assert isinstance(tracker.artifact_path, Path)


def test_noop_exp_tracker_checkpoint_path(tracker: NoOpExpTracker) -> None:
    assert isinstance(tracker.checkpoint_path, Path)


def test_noop_exp_tracker_experiment_id(tracker: NoOpExpTracker) -> None:
    assert tracker.experiment_id == "fakeid0123"


def test_noop_exp_tracker_start_already_activated(tracker: NoOpExpTracker) -> None:
    with raises(RuntimeError):
        tracker.start()


def test_noop_exp_tracker_flush(tracker: NoOpExpTracker) -> None:
    tracker.flush()


def test_noop_exp_tracker_end() -> None:
    tracker = NoOpExpTracker()
    with tracker:
        assert tracker.is_activated()
    assert not tracker.is_activated()
    assert tracker._remove_after_run


def test_noop_exp_tracker_end_remove_after_run_false(tmp_path: Path) -> None:
    tracker = NoOpExpTracker(tmp_path)
    with tracker:
        assert tracker.is_activated()
    assert not tracker.is_activated()
    assert not tracker._remove_after_run


def test_noop_exp_tracker_is_activated_true(tracker: NoOpExpTracker) -> None:
    assert tracker.is_activated()


def test_noop_exp_tracker_is_resumed_false(tracker: NoOpExpTracker) -> None:
    assert not tracker.is_resumed()


@mark.parametrize("value", (1, 1.12, "something"))
def test_noop_exp_tracker_add_tag(tracker: NoOpExpTracker, value: Any) -> None:
    tracker.add_tag("key", value)


def test_noop_exp_tracker_add_tags(tracker: NoOpExpTracker) -> None:
    tracker.add_tags({"int": 1, "float": 1.12, "str": "something"})


def test_noop_exp_tracker_create_artifact(tracker: NoOpExpTracker) -> None:
    tracker.create_artifact(JSONArtifact(tag="metric", data={"f1_score": 42}))
    assert not tracker.artifact_path.joinpath("metric.json").is_file()


def test_noop_exp_tracker_log_best_metric(tracker: NoOpExpTracker) -> None:
    tracker.log_best_metric("key", 1.2)


def test_noop_exp_tracker_log_best_metrics(tracker: NoOpExpTracker) -> None:
    tracker.log_best_metrics({"loss": 1.2, "accuracy": 35})


def test_noop_exp_tracker_log_figure(tracker: NoOpExpTracker) -> None:
    tracker.log_figure("my_figure", Mock(spec=Figure))


def test_noop_exp_tracker_log_figures(tracker: NoOpExpTracker) -> None:
    tracker.log_figures({"my_figure_1": Mock(spec=Figure), "my_figure_2": Mock(spec=Figure)})


@mark.parametrize("value", (1.2, 35, "abc", nn.Linear(4, 5), torch.ones(2, 3)))
def test_noop_exp_tracker_log_hyper_parameter(tracker: NoOpExpTracker, value: Any) -> None:
    tracker.log_hyper_parameter("param", value)


def test_noop_exp_tracker_log_hyper_parameters(tracker: NoOpExpTracker) -> None:
    tracker.log_hyper_parameters(
        {
            "param_float": 1.2,
            "param_int": 35,
            "param_str": "abc",
            "param_nn": nn.Linear(4, 5),
            "param_tensor": torch.ones(2, 3),
        }
    )


def test_noop_exp_tracker_log_image(tracker: NoOpExpTracker) -> None:
    tracker.log_image("my_image", Mock(spec=Image))


def test_noop_exp_tracker_log_images(tracker: NoOpExpTracker) -> None:
    tracker.log_images({"my_image_1": Mock(spec=Image), "my_image_2": Mock(spec=Image)})


@mark.parametrize("value", (1.2, 35))
def test_noop_exp_tracker_log_metric_without_step(
    tracker: NoOpExpTracker, value: Union[int, float]
) -> None:
    tracker.log_metric("key", value)


def test_noop_exp_tracker_log_metric_with_epoch_step(tracker: NoOpExpTracker) -> None:
    tracker.log_metric("key", 1.2, step=EpochStep(2))


def test_noop_exp_tracker_log_metric_with_iteration_step(tracker: NoOpExpTracker) -> None:
    tracker.log_metric("key", 1.2, step=IterationStep(20))


def test_noop_exp_tracker_log_metrics_without_step(tracker: NoOpExpTracker) -> None:
    tracker.log_metrics({"loss": 1.2, "accuracy": 35})


def test_noop_exp_tracker_log_metrics_with_epoch_step(tracker: NoOpExpTracker) -> None:
    tracker.log_metrics({"loss": 1.2, "accuracy": 35}, step=EpochStep(2))


def test_noop_exp_tracker_log_metrics_with_iteration_step(tracker: NoOpExpTracker) -> None:
    tracker.log_metrics({"loss": 1.2, "accuracy": 35}, step=IterationStep(20))


def test_noop_exp_tracker_upload_checkpoints(tracker: NoOpExpTracker) -> None:
    tracker.upload_checkpoints()


def test_noop_exp_tracker_duplicate_start(tracker: NoOpExpTracker) -> None:
    with raises(RuntimeError):
        tracker.start()


def test_noop_exp_tracker_temporary_dir() -> None:
    with NoOpExpTracker() as tracker:
        assert tracker._remove_after_run
        assert tracker._experiment_path.is_dir()


###################################################
#     Tests when the tracker is not activated     #
###################################################


@fixture(scope="module")
def not_activated_tracker(tmp_path_factory: TempPathFactory) -> NoOpExpTracker:
    return NoOpExpTracker(experiment_path=tmp_path_factory.mktemp("data").as_posix())


def test_not_activated_noop_exp_tracker_artifact_path(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.artifact_path


def test_not_activated_noop_exp_tracker_checkpoint_path(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.checkpoint_path


def test_not_activated_noop_exp_tracker_experiment_id(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.experiment_id


def test_not_activated_noop_exp_tracker_flush(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.flush()


def test_not_activated_noop_exp_tracker_end(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.end()


def test_not_activated_noop_exp_tracker_is_activated(not_activated_tracker: NoOpExpTracker) -> None:
    assert not not_activated_tracker.is_activated()


def test_not_activated_noop_exp_tracker_is_resumed(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.is_resumed()


def test_not_activated_noop_exp_tracker_add_tag(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.add_tag("mode", "eval")


def test_not_activated_noop_exp_tracker_add_tags(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.add_tags({"machine": "mac", "mode": "training"})


def test_not_activated_noop_exp_tracker_create_artifact(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.create_artifact(JSONArtifact(tag="metric", data={"f1_score": 42}))


def test_not_activated_noop_exp_tracker_log_best_metric(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_best_metric("my_metric", 12)


def test_not_activated_noop_exp_tracker_log_best_metrics(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_best_metrics({"my_metric_1": 12, "my_metric_2": 3.5})


def test_not_activated_noop_exp_tracker_log_figure(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_figure("my_figure", Mock(spec=Figure))


def test_not_activated_noop_exp_tracker_log_figures(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_figures(
            {"my_figure_1": Mock(spec=Figure), "my_figure_2": Mock(spec=Figure)}
        )


def test_not_activated_noop_exp_tracker_log_hyper_parameter(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_hyper_parameter("model.network.input_size", 12)


def test_not_activated_noop_exp_tracker_log_hyper_parameters(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_hyper_parameters(
            {"model.network.hidden_size": 16, "model.network.dropout": 0.5}
        )


def test_not_activated_noop_exp_tracker_log_image(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_image("my_image", Mock(spec=Image))


def test_not_activated_noop_exp_tracker_log_images(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_images(
            {"my_image_1": Mock(spec=Image), "my_image_2": Mock(spec=Image)}
        )


def test_not_activated_noop_exp_tracker_log_metric(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_metric("my_metric", 0.12)


def test_not_activated_noop_exp_tracker_log_metrics(not_activated_tracker: NoOpExpTracker) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.log_metrics({"accuracy1": 1.12, "accuracy2": 2.12})


def test_not_activated_noop_exp_tracker_upload_checkpoints(
    not_activated_tracker: NoOpExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.upload_checkpoints()
