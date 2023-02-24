r"""This module implements an experiment tracker using TensorBoard."""

__all__ = ["TensorBoardExpTracker"]

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import Mock

import torch

from gravitorch import constants as ct
from gravitorch.utils.artifacts import BaseArtifact
from gravitorch.utils.exp_trackers.base import (
    BaseBasicExpTracker,
    NotActivatedExpTrackerError,
)
from gravitorch.utils.exp_trackers.steps import Step
from gravitorch.utils.format import to_flat_dict
from gravitorch.utils.integrations import (
    check_tensorboard,
    is_matplotlib_available,
    is_pillow_available,
    is_tensorboard_available,
)
from gravitorch.utils.io import load_json, save_json
from gravitorch.utils.path import sanitize_path

if is_matplotlib_available():
    from matplotlib.pyplot import Figure
else:
    Figure = "matplotlib.pyplot.Figure"  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image
else:
    Image = "PIL.Image.Image"  # pragma: no cover

if is_tensorboard_available():
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.tensorboard.summary import hparams
else:
    SummaryWriter, hparams = Mock, None  # pragma: no cover

logger = logging.getLogger(__name__)

BEST_METRIC_FILENAME = "best_metric.json"
MODEL_FILENAME = "model.txt"
TENSORBOARD_FOLDER_NAME = "tb"


class TensorBoardExpTracker(BaseBasicExpTracker):
    r"""Implements an experiment tracker using TensorBoard.

    This module is under development and only supports some features.

    Args:
        experiment_path (``pathlib.Path`` or str or ``None``,
            optional): Specifies the path where to write the
            experiment logs. If ``None``, a temporary directory is
            used. The temporary directory will be removed at the end
            of the experiment. Default: ``None``
        remove_after_run (bool, optional): If ``True``, the experiment
            directory is deleted at the end of the run, otherwise the
            experiment directory is kept. This option is ignored if a
            temporary directory is used. Default: ``False``
    """

    def __init__(
        self,
        experiment_path: Union[Path, str, None] = None,
        remove_after_run: bool = False,
    ):
        check_tensorboard()
        self._experiment_path = sanitize_path(experiment_path) if experiment_path else None
        # Flag to indicate if the tracker is activated or not
        self._is_activated = False
        # Flag to indicate if the experiment directory should be removed at the end of the run.
        self._remove_after_run = bool(remove_after_run)

        # The following directories or filenames are defined in the start method
        self._checkpoint_path: Optional[Path] = None
        self._artifact_path: Optional[Path] = None
        self._best_metric_path: Optional[Path] = None
        # The writer is initialized in the start method
        self._writer = None

        self._best_metrics = {}
        self._hparams = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(experiment_path={self._experiment_path}, "
            f"is_activated={self._is_activated})"
        )

    @property
    def artifact_path(self) -> Path:
        if not self._is_activated:
            raise NotActivatedExpTrackerError(
                "It is not possible to get `artifact_path` because the experiment tracker "
                "is not activated"
            )
        return self._artifact_path

    @property
    def checkpoint_path(self) -> Path:
        if not self._is_activated:
            raise NotActivatedExpTrackerError(
                "It is not possible to get `checkpoint_path` because the experiment tracker "
                "is not activated"
            )
        return self._checkpoint_path

    @property
    def experiment_id(self) -> str:
        if not self._is_activated:
            raise NotActivatedExpTrackerError(
                "It is not possible to get `experiment_id` because the experiment tracker "
                "is not activated"
            )
        return "fakeid0123"

    def _start(self) -> None:
        self._is_activated = True
        if self._experiment_path is None:
            self._experiment_path = sanitize_path(tempfile.mkdtemp())
            self._remove_after_run = True

        logger.info(
            f"Create a TensorBoard experiment tracker and store results at {self._experiment_path}"
        )
        self._best_metric_path = self._experiment_path.joinpath(BEST_METRIC_FILENAME)
        self._writer = MLTorchSummaryWriter(
            log_dir=self._experiment_path.joinpath(TENSORBOARD_FOLDER_NAME).as_posix()
        )
        self._artifact_path = self._experiment_path.joinpath(ct.ARTIFACT_FOLDER_NAME)
        self._checkpoint_path = self._experiment_path.joinpath(ct.CHECKPOINT_FOLDER_NAME)

        self._load_best_metrics()

    def _flush(self, upload_checkpoints: bool = True) -> None:
        self._save_best_metrics()
        self._writer.add_hparams(_sanitize_dict(self._hparams), self._best_metrics)
        self._writer.flush()
        if upload_checkpoints:
            self.upload_checkpoints()

    def _end(self) -> None:
        logger.info("Stopping the TensorBoard experiment tracker...")
        self.flush()
        logger.info(f"::::: TensorBoard logs are available at {self._experiment_path} :::::")
        self._writer.close()
        self._writer = None
        self._best_metrics.clear()
        self._hparams.clear()

        if self._remove_after_run:
            # Remove the temporary directory
            shutil.rmtree(self._experiment_path)
            self._experiment_path = None

        self._artifact_path = None
        self._checkpoint_path = None
        self._is_activated = False
        self._best_metric_path = None

    def is_activated(self) -> bool:
        return self._is_activated

    def _is_resumed(self) -> bool:
        return False  # TODO

    def _add_tag(self, name: str, value: Any) -> None:
        pass  # Do nothing because there is no equivalent feature in TensorBoard

    def _add_tags(self, tags: dict[str, Any]) -> None:
        pass  # Do nothing because there is no equivalent feature in TensorBoard

    def _create_artifact(self, artifact: BaseArtifact) -> None:
        artifact.create(self.artifact_path)

    def _log_best_metric(self, key: str, value: Union[int, float]) -> None:
        self._best_metrics[f"{key}.best"] = _sanitize_value(value)

    def _log_best_metrics(self, metrics: dict[str, Union[int, float]]) -> None:
        for key, value in metrics.items():
            self._log_best_metric(key, value)

    def _log_figure(self, key: str, figure: Figure, step: Optional[Step] = None) -> None:
        self._writer.add_figure(key, figure, step.step if step else step)

    def _log_figures(self, figures: dict[str, Figure], step: Optional[Step] = None) -> None:
        step = step.step if step else step
        for key, figure in figures.items():
            self._writer.add_figure(key, figure, step)

    def _log_hyper_parameter(self, key: str, value: Any) -> None:
        self._hparams[key] = value

    def _log_hyper_parameters(self, params: dict[str, Any]) -> None:
        self._hparams.update(params)

    def _log_image(self, key: str, image: Image, step: Optional[Step] = None) -> None:
        pass  # Do nothing because there is no equivalent feature in TensorBoard

    def _log_images(self, images: dict[str, Image], step: Optional[Step] = None) -> None:
        pass  # Do nothing because there is no equivalent feature in TensorBoard

    def _log_metric(self, key: str, value: Union[int, float], step: Optional[Step] = None) -> None:
        self._writer.add_scalar(key, _sanitize_value(value), step.step if step else step)

    def _log_metrics(
        self, metrics: dict[str, Union[int, float]], step: Optional[Step] = None
    ) -> None:
        step = step.step if step else step
        for key, value in _sanitize_dict(metrics).items():
            self._writer.add_scalar(key, value, step)

    def _upload_checkpoints(self) -> None:
        pass  # Do nothing because it is not possible to upload checkpoints to TensorBoard

    def _load_best_metrics(self) -> None:
        r"""Loads the best metrics from a JSON file."""
        if self._best_metric_path.is_file():
            self._best_metrics = load_json(self._best_metric_path)

    def _save_best_metrics(self) -> None:
        r"""Saves the best metrics in a JSON file."""
        save_json(self._best_metrics, self._best_metric_path)


class MLTorchSummaryWriter(SummaryWriter):
    r"""Implements a variant of the ``SummaryWriter``.

    The goal is to refine ``add_hparams`` to be able to call it several
    times per experiment. Otherwise, a new experiment is created every
    time thath/issues/32651#issuecomment-643791116
    orch/issues/32651#issuecomment-648340103
    https://github.com/pytorch/pyt
    """

    def add_hparams(self, hparam_dict, metric_dict):
        r"""Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


def _sanitize_dict(hparams: dict[str, Any]) -> dict[str, Any]:
    """Computes a sanitized version of the hyper-parameters.

    The values that are not compatible with TensorBoard are converted
    to a string.

    Args:
        hparams (dict): Specifies the hyper-parameters to sanitize.

    Returns:
        dict: The sanitized hyper-parameters.
    """
    return {key: _sanitize_value(value) for key, value in to_flat_dict(hparams).items()}


def _sanitize_value(value: Any) -> Union[bool, int, float, str]:
    """Computes a sanitized a value.

    The values that are not compatible with TensorBoard are converted
    to a string.

    Args:
        value (dict): Specifies the value to sanitize.

    Returns:
        dict: The sanitized value.
    """
    if isinstance(value, (bool, int, float, str)):
        return value
    elif torch.is_tensor(value) and value.numel() == 1:
        return value
    return str(value)
