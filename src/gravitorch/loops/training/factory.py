r"""This module implements some utility functions for the training
loops."""

from __future__ import annotations

__all__ = ["is_training_loop_config", "setup_training_loop"]

import logging

from objectory.utils import is_object_config

from gravitorch.loops.training.base import BaseTrainingLoop
from gravitorch.loops.training.vanilla import TrainingLoop
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_training_loop_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseTrainingLoop``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
    ----
        config (dict): Specifies the configuration to check.

    Returns:
    -------
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseTrainingLoop`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training import is_training_loop_config
        >>> is_training_loop_config({"_target_": "gravitorch.loops.training.TrainingLoop"})
        True
    """
    return is_object_config(config, BaseTrainingLoop)


def setup_training_loop(training_loop: BaseTrainingLoop | dict | None) -> BaseTrainingLoop:
    r"""Sets up the training loop.

    The training loop is instantiated from its configuration by using
    the ``BaseTrainingLoop`` factory function.

    Args:
    ----
        training_loop (``BaseTrainingLoop`` or dict or None):
            Specifies the training loop or its configuration.
            If ``None``, the ``TrainingLoop`` is instantiated.

    Returns:
    -------
        ``BaseTrainingLoop``: The training loop.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training import setup_training_loop
        >>> loop = setup_training_loop({"_target_": "gravitorch.loops.training.TrainingLoop"})
        >>> loop
        TrainingLoop(
          (set_grad_to_none): True
          (batch_device_placement): AutoDevicePlacement(device=cpu)
          (tag): train
          (clip_grad_fn): None
          (clip_grad_args): ()
          (observer): NoOpLoopObserver()
          (profiler): NoOpProfiler()
        )
    """
    if training_loop is None:
        training_loop = TrainingLoop()
    if isinstance(training_loop, dict):
        logger.info(
            "Initializing a training loop from its configuration... "
            f"{str_target_object(training_loop)}"
        )
        training_loop = BaseTrainingLoop.factory(**training_loop)
    if not isinstance(training_loop, BaseTrainingLoop):
        logger.warning(
            f"training_loop is not a `BaseTrainingLoop` (received: {type(training_loop)})"
        )
    return training_loop
