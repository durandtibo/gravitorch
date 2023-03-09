from pathlib import Path
from typing import Any, Union
from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from pytest import TempPathFactory, fixture, mark, raises
from torch import nn

from gravitorch.testing import tensorboard_available
from gravitorch.utils.artifacts import JSONArtifact
from gravitorch.utils.exp_trackers import (
    EpochStep,
    IterationStep,
    NotActivatedExpTrackerError,
)
from gravitorch.utils.exp_trackers.tensorboard import (
    MLTorchSummaryWriter,
    TensorBoardExpTracker,
    _sanitize_dict,
)
from gravitorch.utils.integrations import is_matplotlib_available, is_pillow_available
from gravitorch.utils.io import save_json

if is_matplotlib_available():
    from matplotlib.pyplot import Figure
else:
    Figure = "matplotlib.pyplot.Figure"  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image
else:
    Image = "PIL.Image.Image"  # pragma: no cover


###########################################
#     Tests for TensorBoardExpTracker     #
###########################################


@tensorboard_available
def test_tensorboard_exp_tracker_str() -> None:
    assert str(TensorBoardExpTracker()).startswith("TensorBoardExpTracker(")


@tensorboard_available
def test_tensorboard_exp_tracker_artifact_path(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        assert isinstance(tracker.artifact_path, Path)


@tensorboard_available
def test_tensorboard_exp_tracker_checkpoint_path(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        assert isinstance(tracker.checkpoint_path, Path)


@tensorboard_available
def test_tensorboard_exp_tracker_experiment_id(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        assert tracker.experiment_id == "fakeid0123"


@tensorboard_available
@mark.parametrize("upload_checkpoints", (True, False))
def test_tensorboard_exp_tracker_flush(tmp_path: Path, upload_checkpoints: bool) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.flush(upload_checkpoints)
        assert tracker._best_metric_path.is_file()


@tensorboard_available
@mark.parametrize("remove_after_run", (True, False))
def test_tensorboard_exp_tracker_end(tmp_path: Path, remove_after_run: bool) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ):
        tracker = TensorBoardExpTracker(experiment_path=tmp_path, remove_after_run=remove_after_run)
        with tracker:
            assert tracker.is_activated()
        assert not tracker.is_activated()
        assert tracker._remove_after_run == remove_after_run


@tensorboard_available
def test_tensorboard_exp_tracker_is_activated_true(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        assert tracker.is_activated()


@tensorboard_available
def test_tensorboard_exp_tracker_is_resumed_false(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        assert not tracker.is_resumed()


#############################
#     Tests for add_tag     #
#############################
@tensorboard_available
@mark.parametrize("value", (1, 1.12, "something"))
def test_tensorboard_exp_tracker_add_tag(tmp_path: Path, value: Any) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.add_tag("key", value)


@tensorboard_available
def test_tensorboard_exp_tracker_add_tags(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.add_tags({"int": 1, "float": 1.12, "str": "something"})


@tensorboard_available
def test_tensorboard_exp_tracker_create_artifact(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.create_artifact(JSONArtifact(tag="metric", data={"f1_score": 42}))
        assert tracker.artifact_path.joinpath("metric.json").is_file()


#####################################
#     Tests for log_best_metric     #
#####################################


@tensorboard_available
def test_tensorboard_exp_tracker_log_best_metric(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_best_metric("key", 1.2)
        assert tracker._best_metrics["key.best"] == 1.2


@tensorboard_available
def test_tensorboard_exp_tracker_log_best_metrics(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_best_metrics({"loss": 1.2, "accuracy": 35})
        assert tracker._best_metrics["loss.best"] == 1.2
        assert tracker._best_metrics["accuracy.best"] == 35


################################
#     Tests for log_figure     #
################################


@tensorboard_available
def test_tensorboard_exp_tracker_log_figure_without_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figure("my_figure", figure)
        writer.add_figure.assert_called_once_with("my_figure", figure, None)


@tensorboard_available
def test_tensorboard_exp_tracker_log_figure_with_epoch_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figure("my_figure_epoch", figure, EpochStep(2))
        writer.add_figure.assert_called_once_with("my_figure_epoch", figure, 2)


@tensorboard_available
def test_tensorboard_exp_tracker_log_figure_with_iteration_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figure("my_figure_iteration", figure, IterationStep(20))
        writer.add_figure.assert_called_once_with("my_figure_iteration", figure, 20)


@tensorboard_available
def test_tensorboard_exp_tracker_log_figures_without_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figures({"my_figure_1": figure, "my_figure_2": figure})
        assert writer.add_figure.call_args_list == [
            (("my_figure_1", figure, None), {}),
            (("my_figure_2", figure, None), {}),
        ]


@tensorboard_available
def test_tensorboard_exp_tracker_log_figures_with_epoch_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figures(
            {"my_figure_epoch_1": figure, "my_figure_epoch_2": figure}, EpochStep(2)
        )
        assert writer.add_figure.call_args_list == [
            (("my_figure_epoch_1", figure, 2), {}),
            (("my_figure_epoch_2", figure, 2), {}),
        ]


@tensorboard_available
def test_tensorboard_exp_tracker_log_figures_with_iteration_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        figure = Mock(spec=Figure)
        tracker.log_figures(
            {"my_figure_iteration_1": figure, "my_figure_iteration_2": figure},
            IterationStep(28),
        )
        assert writer.add_figure.call_args_list == [
            (("my_figure_iteration_1", figure, 28), {}),
            (("my_figure_iteration_2", figure, 28), {}),
        ]


#########################################
#     Tests for log_hyper_parameter     #
#########################################


@tensorboard_available
@mark.parametrize("value", (1.2, 35, "abc", nn.Linear(4, 5), torch.ones(2), None))
def test_tensorboard_exp_tracker_log_hyper_parameter(tmp_path: Path, value: Any) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_hyper_parameter("param", value)
        assert objects_are_equal(tracker._hparams["param"], value)


@tensorboard_available
def test_tensorboard_exp_tracker_log_hyper_parameters(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_hyper_parameters(
            {
                "param_float": 1.2,
                "param_int": 35,
                "param_str": "abc",
                "param_nn": nn.Linear(4, 5),
                "param_tensor": torch.ones(1, 3),
                "param_none": None,
            }
        )
        assert tracker._hparams["param_float"] == 1.2
        assert tracker._hparams["param_int"] == 35
        assert tracker._hparams["param_str"] == "abc"
        assert isinstance(tracker._hparams["param_nn"], nn.Linear)
        assert tracker._hparams["param_tensor"].equal(torch.ones(1, 3))
        assert tracker._hparams["param_none"] is None


###############################
#     Tests for log_image     #
###############################


@tensorboard_available
def test_tensorboard_exp_tracker_log_image(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_image("my_image", Mock(spec=Image))


@tensorboard_available
def test_tensorboard_exp_tracker_log_image_with_epoch_step(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_image("my_image_epoch", Mock(spec=Image), EpochStep(2))


@tensorboard_available
def test_tensorboard_exp_tracker_log_image_with_iteration_step(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_image("my_image_iteration", Mock(spec=Image), IterationStep(20))


@tensorboard_available
def test_tensorboard_exp_tracker_log_images(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_images({"my_image_1": Mock(spec=Image), "my_image_2": Mock(spec=Image)})


@tensorboard_available
def test_tensorboard_exp_tracker_log_images_with_epoch_step(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_images(
            {
                "my_image_epoch_1": Mock(spec=Image),
                "my_image_epoch_2": Mock(spec=Image),
            },
            EpochStep(2),
        )


@tensorboard_available
def test_tensorboard_exp_tracker_log_images_with_iteration_step(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_images(
            {
                "my_image_iteration_1": Mock(spec=Image),
                "my_image_iteration_2": Mock(spec=Image),
            },
            IterationStep(20),
        )


################################
#     Tests for log_metric     #
################################


@tensorboard_available
@mark.parametrize("value", (1.2, 35))
def test_tensorboard_exp_tracker_log_metric_without_step(
    tmp_path: Path, value: Union[int, float]
) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metric("key", value)
        writer.add_scalar.assert_called_once_with("key", value, None)


@tensorboard_available
def test_tensorboard_exp_tracker_log_metric_with_epoch_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metric("key", 1.2, step=EpochStep(2))
        writer.add_scalar.assert_called_once_with("key", 1.2, 2)


@tensorboard_available
def test_tensorboard_exp_tracker_log_metric_with_iteration_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metric("key", 1.2, step=IterationStep(20))
        writer.add_scalar.assert_called_once_with("key", 1.2, 20)


@tensorboard_available
def test_tensorboard_exp_tracker_log_metrics_without_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metrics({"loss": 1.2, "accuracy": 35})
        assert writer.add_scalar.call_args_list == [
            (("loss", 1.2, None), {}),
            (("accuracy", 35, None), {}),
        ]


@tensorboard_available
def test_tensorboard_exp_tracker_log_metrics_with_epoch_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metrics({"loss": 1.2, "accuracy": 35}, step=EpochStep(2))
        assert writer.add_scalar.call_args_list == [
            (("loss", 1.2, 2), {}),
            (("accuracy", 35, 2), {}),
        ]


@tensorboard_available
def test_tensorboard_exp_tracker_log_metrics_with_iteration_step(tmp_path: Path) -> None:
    writer = Mock()
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=writer)
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker.log_metrics({"loss": 1.2, "accuracy": 35}, step=IterationStep(20))
        assert writer.add_scalar.call_args_list == [
            (("loss", 1.2, 20), {}),
            (("accuracy", 35, 20), {}),
        ]


@tensorboard_available
def test_tensorboard_exp_tracker_duplicate_start(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker, raises(RuntimeError):
        tracker.start()


@tensorboard_available
def test_tensorboard_exp_tracker_temporary_dir() -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker() as tracker:
        assert tracker._remove_after_run
        assert tracker._experiment_path.is_dir()


@tensorboard_available
def test_tensorboard_exp_tracker_clean_internal_variables() -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ):
        tracker = TensorBoardExpTracker()
        with tracker:
            assert tracker.is_activated()
        assert not tracker.is_activated()

        assert tracker._best_metrics == {}
        assert tracker._hparams == {}
        assert tracker._experiment_path is None
        assert tracker._artifact_path is None
        assert tracker._checkpoint_path is None
        assert tracker._best_metric_path is None


@tensorboard_available
def test_tensorboard_exp_tracker_load_best_metrics(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        save_json({"accuracy": 45, "loss": 1.2}, tracker._best_metric_path)
        tracker._load_best_metrics()
        assert tracker._best_metrics == {"accuracy": 45, "loss": 1.2}


@tensorboard_available
def test_tensorboard_exp_tracker_load_best_metrics_no_file(tmp_path: Path) -> None:
    with patch(
        "gravitorch.utils.exp_trackers.tensorboard.MLTorchSummaryWriter", Mock(return_value=Mock())
    ), TensorBoardExpTracker(experiment_path=tmp_path) as tracker:
        tracker._load_best_metrics()
        assert tracker._best_metrics == {}


###################################################
#     Tests when the tracker is not activated     #
###################################################


@fixture(scope="module")
def not_activated_tracker(tmp_path_factory: TempPathFactory) -> TensorBoardExpTracker:
    return TensorBoardExpTracker(experiment_path=tmp_path_factory.mktemp("data").as_posix())


@tensorboard_available
def test_tensorboard_exp_tracker_not_activated_artifact_dir(
    not_activated_tracker: TensorBoardExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.artifact_path


@tensorboard_available
def test_tensorboard_exp_tracker_not_activated_checkpoint_dir(
    not_activated_tracker: TensorBoardExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.checkpoint_path


@tensorboard_available
def test_tensorboard_exp_tracker_not_activated_experiment_id(
    not_activated_tracker: TensorBoardExpTracker,
) -> None:
    with raises(NotActivatedExpTrackerError):
        not_activated_tracker.experiment_id


@tensorboard_available
def test_tensorboard_exp_tracker_not_activated_is_activated(
    not_activated_tracker: TensorBoardExpTracker,
) -> None:
    assert not not_activated_tracker.is_activated()


##########################################
#     Tests for MLTorchSummaryWriter     #
##########################################


@tensorboard_available
def test_gravitorch_summary_writer(tmp_path: Path) -> None:
    with MLTorchSummaryWriter(tmp_path.as_posix()) as writer:
        writer.add_hparams(
            hparam_dict={"param1": 1, "param2": 2}, metric_dict={"metric1": 10, "metric2": 20}
        )


@tensorboard_available
def test_gravitorch_summary_writer_incorrect_type(tmp_path: Path) -> None:
    with MLTorchSummaryWriter(tmp_path.as_posix()) as writer, raises(TypeError):
        writer.add_hparams(hparam_dict=Mock(), metric_dict=Mock())


####################################
#     Tests for _sanitize_dict     #
####################################


def test_sanitize_dict_flat_dict() -> None:
    assert _sanitize_dict({"model": {"network": "resnet18", "metric": "mse"}}) == {
        "model.network": "resnet18",
        "model.metric": "mse",
    }


def test_sanitize_dict_types() -> None:
    assert objects_are_equal(
        _sanitize_dict(
            {
                "bool": True,
                "int": 1,
                "float": 2.5,
                "str": "abc",
                "tensor_1": torch.ones(1),
                "tensor_2": torch.ones(1, 3),
                "module": nn.Linear(4, 6),
                "none": None,
            }
        ),
        {
            "bool": True,
            "int": 1,
            "float": 2.5,
            "str": "abc",
            "tensor_1": torch.ones(1),
            "tensor_2": "tensor([[1., 1., 1.]])",
            "module": "Linear(in_features=4, out_features=6, bias=True)",
            "none": "None",
        },
    )
