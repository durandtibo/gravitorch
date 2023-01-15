r"""This module implements a no-operation experiment tracker."""

__all__ = ["NoOpExpTracker"]

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from gravitorch import constants as ct
from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.exp_trackers.base import (
    BaseBasicExpTracker,
    NotActivatedExpTrackerError,
)
from gravitorch.utils.exp_trackers.steps import Step
from gravitorch.utils.integrations import is_matplotlib_available, is_pillow_available
from gravitorch.utils.path import sanitize_path

if is_matplotlib_available():
    from matplotlib.pyplot import Figure
else:
    Figure = "matplotlib.pyplot.Figure"  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image
else:
    Image = "PIL.Image.Image"  # pragma: no cover

logger = logging.getLogger(__name__)


class NoOpExpTracker(BaseBasicExpTracker):
    r"""Implements a no-operation experiment tracker.

    This class is equivalent to disable the experiment tracker without
    changing the current implementation. You should not use this class
    if you want to track the experiment artifacts. This class creates
    the paths, so you can write files like for the other experiment
    trackers.

    Args:
        experiment_path (``pathlib.Path`` or str or ``None``,
            optional): Specifies the path where to write the
            experiment logs. If ``None``, a temporary directory is
            used. The temporary directory will be removed at the end
            of the experiment. Default: ``None``
    """

    def __init__(self, experiment_path: Union[Path, str, None] = None):
        self._experiment_path = sanitize_path(experiment_path) if experiment_path else None
        # Flag to indicate if the tracker is activated or not
        self._is_activated = False
        # Flag to indicate if the experiment directory should be removed at the end of the run
        self._remove_after_run = False

        # The following directories are defined when the start method is called
        self._checkpoint_path: Optional[Path] = None
        self._artifact_path: Optional[Path] = None

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
            f"Create a noop experiment tracker and store results at {self._experiment_path}"
        )
        self._artifact_path = self._experiment_path.joinpath(ct.ARTIFACT_FOLDER_NAME)
        self._checkpoint_path = self._experiment_path.joinpath(ct.CHECKPOINT_FOLDER_NAME)

    def _flush(self, upload_checkpoints: bool = True) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _end(self) -> None:
        if self._remove_after_run:
            # Remove the temporary directory
            shutil.rmtree(self._experiment_path)
            self._experiment_path = None

        self._artifact_path = None
        self._checkpoint_path = None
        self._is_activated = False

    def is_activated(self) -> bool:
        return self._is_activated

    def _is_resumed(self) -> bool:
        return False  # Always none because this experiment tracker cannot be resumed.

    def _add_tag(self, name: str, value: Any) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _add_tags(self, tags: dict[str, Any]) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _create_artifact(self, artifact: BaseArtifact) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_best_metric(self, key: str, value: Union[int, float]) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_best_metrics(self, metrics: dict[str, Union[int, float]]) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_figure(self, key: str, figure: Figure, step: Optional[Step] = None) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_figures(self, figures: dict[str, Figure], step: Optional[Step] = None) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_hyper_parameter(self, key: str, value: Any) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_hyper_parameters(self, params: dict[str, Any]) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_image(self, key: str, image: Image, step: Optional[Step] = None) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_images(self, images: dict[str, Image], step: Optional[Step] = None) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_metric(self, key: str, value: Union[int, float], step: Optional[Step] = None) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _log_metrics(
        self, metrics: dict[str, Union[int, float]], step: Optional[Step] = None
    ) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.

    def _upload_checkpoints(self) -> None:
        pass  # Do nothing because it is a no-operation experiment tracker.
