r"""This module defines the base class for the experiment trackers."""

__all__ = [
    "BaseBasicExpTracker",
    "BaseExpTracker",
    "NotActivatedExpTrackerError",
]

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from objectory import AbstractFactory

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.exp_trackers.steps import Step
from gravitorch.utils.integrations import is_matplotlib_available, is_pillow_available

if is_matplotlib_available():
    from matplotlib.pyplot import Figure
else:
    Figure = "matplotlib.pyplot.Figure"  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image
else:
    Image = "PIL.Image.Image"  # pragma: no cover


class BaseExpTracker(ABC, metaclass=AbstractFactory):
    r"""Defines the base class to implement an experiment tracker.

    It is recommended to use the experiment tracker in a context
    manager. Using a context manager ensures the ``start`` and ``end``
    methods are called properly. It is possible to manually call the
    ``start`` and ``end`` methods. It is recommended to not mix these
    two approaches to avoid unexpected side-effects.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.exp_trackers import BaseExpTracker
        # Context manager approach (recommended)
        >>> my_exp_tracker: BaseExpTracker = ...  # Initialize an experiment tracker object
        >>> print(my_exp_tracker.is_activated())
        False
        >>> with my_exp_tracker as exp_tracker:
        ...     print(exp_tracker.is_activated())
        True
        >>> print(my_exp_tracker.is_activated())
        False

        # Manual approach
        >>> exp_tracker: BaseExpTracker = ...  # Initialize an experiment tracker object
        >>> print(exp_tracker.is_activated())
        False
        >>> exp_tracker.start()
        >>> print(exp_tracker.is_activated())
        True
        >>> exp_tracker.end()
        >>> print(exp_tracker.is_activated())
        False
    """

    def __enter__(self) -> "BaseExpTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    @property
    @abstractmethod
    def artifact_path(self) -> Path:
        r"""Gets the path to the artifacts.

        Returns:
            ``pathlib.Path``: The path where you can write some
                artifacts related to the experiment.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.
        """

    @property
    @abstractmethod
    def checkpoint_path(self) -> Path:
        r"""Gets the path to the checkpoints.

        Returns:
            ``pathlib.Path``: The path where you can write the
                checkpoints.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.
        """

    @property
    @abstractmethod
    def experiment_id(self) -> str:
        r"""Gets the experiment ID.

        Returns:
            str: The experiment ID.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.
        """

    @abstractmethod
    def start(self) -> None:
        r"""Starts a new experiment tracking."""

    @abstractmethod
    def flush(self, upload_checkpoints: bool = True) -> None:
        r"""Flushes all the current artifacts/metrics to the remote server.

        This function is expected to be called not at a high frequency.
        For example, you can call this function at the end of each
        epoch.

        Args:
            upload_checkpoints (bool): Indicates if the checkpoints
                are uploaded or not when this method is called.
                Default: ``True``

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.
        """

    @abstractmethod
    def end(self) -> None:
        r"""Ends the tracking of the current experiment.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.
        """

    @abstractmethod
    def is_activated(self) -> bool:
        r"""Indicates if the tracker is activated or not.

        Returns:
            bool: ``True`` if the tracker is activated, otherwise
                ``False``

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.is_activated()
            True
        """

    @abstractmethod
    def is_resumed(self) -> bool:
        r"""Indicates if the experiment was resumed or not.

        Returns:
            bool: ``True`` if the experiment was resumed, otherwise
                ``False``

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.is_resumed()
        """

    @abstractmethod
    def add_tag(self, name: str, value: Any) -> None:
        r"""Adds a tag to the experiment.

        Args:
            name (str): Specifies the name of the tag.
            value: Specifies the value of the tag. The value should be
                convertibled to a string  with the ``str`` function.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.add_tag('mode', 'training')
        """

    @abstractmethod
    def add_tags(self, tags: dict[str, Any]) -> None:
        r"""Adds tags to the experiment.

        Args:
            tags (dict): Specifies the tags to add to the experiment.
                The value should be convertibled to a string with the
                ``str`` function.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.add_tags({'mode': 'training', 'machine': 'mac'})
        """

    @abstractmethod
    def create_artifact(self, artifact: BaseArtifact) -> None:
        r"""Creates an artifact.

        Args:
            artifact (``BaseArtifact``): Specifies the artifact to
                create.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> from gravitorch.utils.artifacts import JSONArtifact
            >>> exp_tracker.create_artifact(JSONArtifact(tag='metric', data={'f1_score': 42}))
        """

    @abstractmethod
    def log_best_metric(self, key: str, value: Any) -> None:
        r"""Logs the best value of a metric.

        Args:
            key (str): Specifies the key used to identify the metric.
                Please do not use the names `'epoch'`, `'iteration'`
                and `'step'` because they are reserved for specific
                uses.
            value (int or float): Specifies the best value to log.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_best_metric('my_metric', 1.2)
        """

    @abstractmethod
    def log_best_metrics(self, metrics: dict[str, Any]) -> None:
        r"""Logs a dictionary of the best metrics.

        Args:
            metrics (dict): Specifies the dictionary of the best
                metrics to log. Please do not use the keys
                ``'epoch'``,  ``'iteration'`` and ``'step'``
                because they are reserved for specific uses.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_best_metrics({"my_metric_1": 12, "my_metric_2": 3.5})
        """

    @abstractmethod
    def log_figure(self, key: str, figure: Figure, step: Optional[Step] = None) -> None:
        r"""Logs a figure for the given key and step.

        Args:
            key (str): Specifies the key used to identify the figure.
            figure (``matplotlib.pyplot.Figure``): Specifies the
                figure to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> import matplotlib.pyplot as plt
            >>> fig, axes = plt.subplots()
            >>> exp_tracker.log_figure('my_figure', fig)  # without step
            >>> from gravitorch.utils.exp_trackers import EpochStep
            >>> exp_tracker.log_figure('my_figure', fig, EpochStep(2))  # with step
        """

    @abstractmethod
    def log_figures(self, figures: dict[str, Figure], step: Optional[Step] = None) -> None:
        r"""Logs a dictionary of figures for a given step.

        Args:
            figures (dict): Specifies the dictionary of figures to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> import matplotlib.pyplot as plt
            >>> fig, axes = plt.subplots()
            >>> exp_tracker.log_figures({'my_figure_1': fig, 'my_figure_2': fig})  # without step
            >>> from gravitorch.utils.exp_trackers import EpochStep
            >>> exp_tracker.log_figures({'my_figure_1': fig, 'my_figure_2': fig}, EpochStep(2))  # with step
        """

    @abstractmethod
    def log_hyper_parameter(self, key: str, value: Any) -> None:
        r"""Logs a single hyper-parameter.

        Args:
            key (str): Specifies the name of the hyper-parameter.
            value (str): Specifies the value of the hyper-parameter.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_hyper_parameter('my_hparam', 32)
        """

    @abstractmethod
    def log_hyper_parameters(self, params: dict[str, Any]) -> None:
        r"""Logs a dictionary of multiple hyper-parameters.

        Args:
            params (dict): Specifies the hyper-parameters to log.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_hyper_parameters({'input_size': 32, 'output_size': 16})
        """

    @abstractmethod
    def log_image(self, key: str, image: Image, step: Optional[Step] = None) -> None:
        r"""Logs an image for the given key and step.

        Args:
            key (str): Specifies the key used to identify the image.
            image (``PIL.Image.Image``): Specifies the image to log.
            step (``Step``, optional): Specifies the step value to
                record. Default: ``None``.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> from PIL import Image
            >>> import numpy as np
            >>> img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), 'RGB')
            >>> exp_tracker.log_image('my_image', img)  # without step
            >>> from gravitorch.utils.exp_trackers import EpochStep
            >>> exp_tracker.log_image('my_image', img, EpochStep(2))  # with step
        """

    @abstractmethod
    def log_images(self, images: dict[str, Image], step: Optional[Step] = None) -> None:
        r"""Logs a dictionary of images for a given step.

        Args:
            images (dict): Specifies the dictionary of images to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> from PIL import Image
            >>> import numpy as np
            >>> img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), 'RGB')
            >>> exp_tracker.log_images({'my_image_1': img, 'my_image_2': img})  # without step
            >>> from gravitorch.utils.exp_trackers.steps import EpochStep
            >>> exp_tracker.log_images({'my_image_1': img, 'my_image_2': img}, EpochStep(2))  # with step
        """

    @abstractmethod
    def log_metric(self, key: str, value: Union[int, float], step: Optional[Step] = None) -> None:
        r"""Logs a single metric.

        Args:
            key (str): Specifies the key used to identify the metric.
                Please do not use the names ``'epoch'``,
                ``'iteration'`` and ``'step'`` because they are
                reserved for specific uses.
            value (int or float): Specifies the value to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_metric('my_metric', 1.2)  # without step
            >>> from gravitorch.utils.exp_trackers import EpochStep
            >>> exp_tracker.log_metric('my_metric', 1.2, EpochStep(2))  # with step
        """

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, Union[int, float]], step: Optional[Step] = None
    ) -> None:
        r"""Logs a dictionary of multiple metrics.

        Args:
            metrics (dict): Specifies the dictionary of metrics to log.
                Please do not use the keys ``'epoch'``, ``'iteration'``
                and ``'step'`` because they are reserved for specific
                uses.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.

        Raises:
            ``NotActivatedExpTrackerError`` if the experiment tracker
                is not activated.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.log_metrics({'my_metric_1': 1.2, 'my_metric_2': 42})  # without step
            >>> from gravitorch.utils.exp_trackers import EpochStep
            >>> exp_tracker.log_metrics({'my_metric_1': 1.2, 'my_metric_2': 42}, EpochStep(2))  # with step
        """

    @abstractmethod
    def upload_checkpoints(self) -> None:
        r"""Uploads all the checkpoints that are in the checkpoint path to the
        remote server.

        This method does nothing if no remote server is used.

        Example usage:

        .. code-block:: python

            >>> from gravitorch.utils.exp_trackers import BaseExpTracker
            >>> exp_tracker: BaseExpTracker = ...  # Initialize and activate the experiment tracker
            >>> exp_tracker.upload_checkpoints()
        """


class NotActivatedExpTrackerError(Exception):
    r"""This exception is raised when you try to do an action with an
    experiment tracker which is not activated."""


class BaseBasicExpTracker(BaseExpTracker):
    r"""Defines a base class to implement some experiment trackers while
    avoiding duplication for some methods.

    This base class implements some basic functionalities to avoid
    duplicate code e.g. code to raise an error if the experiment
    tracker is not implemented.

    Note: it is not necessary to use this base class to implement an
    experiment tracker.
    """

    def start(self) -> None:
        if self.is_activated():
            raise RuntimeError("The experiment tracker is already activated")
        self._start()

    @abstractmethod
    def _start(self) -> None:
        r"""Starts a new experiment tracking."""

    def flush(self, upload_checkpoints: bool = True) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to create flush data because the experiment tracker is not activated"
            )
        self._flush(upload_checkpoints)

    @abstractmethod
    def _flush(self, upload_checkpoints: bool = True) -> None:
        r"""Flushes all the current artifacts/metrics to the remote server.

        This function is expected to be called not at a high frequency.
        For example, you can call this function at the end of each
        epoch.

        Args:
            upload_checkpoints (bool, optional): Indicates if the
                checkpoints are uploaded or not when this method is
                called. Default: ``True``
        """

    def end(self) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to end the tracking because the experiment tracker "
                "is not activated"
            )
        self._end()

    def _end(self) -> None:
        r"""Ends the tracking of the current experiment."""

    def is_resumed(self) -> bool:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to indicate if the experiment was resumed "
                "or not because the experiment tracker is not activated"
            )
        return self._is_resumed()

    @abstractmethod
    def _is_resumed(self) -> bool:
        r"""Indicates if the experiment was resumed or not.

        Returns:
            bool: ``True`` if the experiment was resumed, otherwise
                ``False``.
        """

    def add_tag(self, name: str, value: Any) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to create add the tag {name} because the experiment "
                "tracker is not activated"
            )
        self._add_tag(name, value)

    @abstractmethod
    def _add_tag(self, name: str, value: Any) -> None:
        r"""Adds a tag to the experiment.

        Args:
            name (str): Specifies the name of the tag.
            value: Specifies the value of the tag. The value should be
                convertible to a string with the ``str`` function.
        """

    def add_tags(self, tags: dict[str, Any]) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to create add tags because the experiment tracker is "
                "not activated"
            )
        self._add_tags(tags)

    @abstractmethod
    def _add_tags(self, tags: dict[str, Any]) -> None:
        r"""Adds tags to the experiment.

        Args:
            tags (dict): Specifies the tags to add to the experiment.
                The value should be convertible to a string with the
                ``str`` function.
        """

    def create_artifact(self, artifact: BaseArtifact) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to create the artifact {artifact} because "
                "the experiment tracker is not activated."
            )
        self._create_artifact(artifact)

    @abstractmethod
    def _create_artifact(self, artifact: BaseArtifact) -> None:
        r"""Creates an artifact.

        Args:
            artifact (``BaseArtifact``): Specifies the artifact to
                create.
        """

    def log_best_metric(self, key: str, value: Any) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to log the best metric for {key} because "
                "the experiment tracker is not activated."
            )
        self._log_best_metric(key, value)

    @abstractmethod
    def _log_best_metric(self, key: str, value: Any) -> None:
        r"""Logs the best value of a metric.

        Args:
            key (str): Specifies the key used to identify the metric.
                Please do not use the names ``'epoch'``,
                ``'iteration'`` and ``'step'`` because they are
                reserved for specific uses.
            value (int or float): Specifies the best value to log.
        """

    def log_best_metrics(self, metrics: dict[str, Any]) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to log the best metrics because the experiment tracker "
                "is not activated"
            )
        self._log_best_metrics(metrics)

    @abstractmethod
    def _log_best_metrics(self, metrics: dict[str, Any]) -> None:
        r"""Logs a dictionary of the best metrics.

        Args:
            metrics (dict): Specifies the dictionary of the best
                metrics to log. Please do not use the keys ``'epoch'``,
                ``'iteration'``, and ``'step'`` because they are
                reserved for specific uses.
        """

    def log_figure(self, key: str, figure: Figure, step: Optional[Step] = None) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to log the figure {key} because the experiment tracker "
                "is not activated"
            )
        self._log_figure(key, figure, step)

    @abstractmethod
    def _log_figure(self, key: str, figure: Figure, step: Optional[Step] = None) -> None:
        r"""Logs a figure for the given key and step.

        Args:
            key (str): Specifies the key used to identify the figure.
            figure (``matplotlib.pyplot.Figure``): Specifies the
                figure to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``
        """

    def log_figures(self, figures: dict[str, Figure], step: Optional[Step] = None) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to log the figures because the experiment tracker "
                "is not activated"
            )
        self._log_figures(figures, step)

    @abstractmethod
    def _log_figures(self, figures: dict[str, Figure], step: Optional[Step] = None) -> None:
        r"""Logs a dictionary of figures for a given step.

        Args:
            figures (dict): Specifies the dictionary of figures to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``
        """

    def log_hyper_parameter(self, key: str, value: Any) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to log the hyper-parameter {key} because the "
                "experiment tracker is not activated"
            )
        self._log_hyper_parameter(key, value)

    @abstractmethod
    def _log_hyper_parameter(self, key: str, value: Any) -> None:
        r"""Logs a single hyper-parameter.

        Args:
            key (str): Specifies the name of the hyper-parameter.
            value (str): Specifies the value of the hyper-parameter.
        """

    def log_hyper_parameters(self, params: dict[str, Any]) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to log the hyper-parameters because the experiment tracker "
                "is not activated"
            )
        self._log_hyper_parameters(params)

    @abstractmethod
    def _log_hyper_parameters(self, params: dict[str, Any]) -> None:
        r"""Logs a dictionary of multiple hyper-parameters.

        Args:
            params (dict): Specifies the hyper-parameters to log.
        """

    def log_image(
        self,
        key: str,
        image: Image,  # noqa: F821
        step: Optional[Step] = None,
    ) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to log the image {key} because the experiment tracker "
                "is not activated"
            )
        self._log_image(key, image, step)

    @abstractmethod
    def _log_image(
        self,
        key: str,
        image: Image,  # noqa: F821
        step: Optional[Step] = None,
    ) -> None:
        r"""Logs an image for the given key and step.

        Args:
            key (str): Specifies the key used to identify the image.
            image (``PIL.Image.Image``): Specifies the image to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.
        """

    def log_images(self, images: dict[str, Image], step: Optional[Step] = None) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to log the images because the experiment tracker "
                "is not activated"
            )
        self._log_images(images, step)

    @abstractmethod
    def _log_images(self, images: dict[str, Image], step: Optional[Step] = None) -> None:
        r"""Logs a dictionary of images for a given step.

        Args:
            images (dict): Specifies the dictionary of images to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.
        """

    def log_metric(self, key: str, value: Union[int, float], step: Optional[Step] = None) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                f"It is not possible to log metric {key} because the experiment tracker "
                "is not activated"
            )
        self._log_metric(key, value, step)

    @abstractmethod
    def _log_metric(self, key: str, value: Union[int, float], step: Optional[Step] = None) -> None:
        r"""Logs a single metric.

        Args:
            key (str): Specifies the key used to identify the metric.
                Please do not use the names ``'epoch'``,
                ``'iteration'`` and ``'step'`` because they are
                reserved for specific uses.
            value (int or float): Specifies the value to log.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.
        """

    def log_metrics(
        self, metrics: dict[str, Union[int, float]], step: Optional[Step] = None
    ) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to log metrics because the experiment tracker is not activated"
            )
        self._log_metrics(metrics, step)

    @abstractmethod
    def _log_metrics(
        self, metrics: dict[str, Union[int, float]], step: Optional[Step] = None
    ) -> None:
        r"""Logs a dictionary of multiple metrics.

        Args:
            metrics (dict): Specifies the dictionary of metrics to log.
                Please do not use the keys ``'epoch'``,
                ``'iteration'`` and ``'step'`` because they are
                reserved for specific uses.
            step (``Step``, optional): Specifies the step value to
                record. If ``None``, it will use the default step.
                Default: ``None``.
        """

    def upload_checkpoints(self) -> None:
        if not self.is_activated():
            raise NotActivatedExpTrackerError(
                "It is not possible to upload checkpoints because the experiment tracker "
                "is not activated"
            )
        self._upload_checkpoints()

    @abstractmethod
    def _upload_checkpoints(self) -> None:
        r"""Uploads all the checkpoints to the remote server.

        This method does nothing if no remote server is used.
        """
