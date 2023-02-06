r"""This module implements some utility functions for the training loops."""

__all__ = ["setup_training_loop"]

import logging
from typing import Union

from gravitorch.loops.training.base import BaseTrainingLoop
from gravitorch.loops.training.vanilla import VanillaTrainingLoop
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_training_loop(training_loop: Union[BaseTrainingLoop, dict, None]) -> BaseTrainingLoop:
    r"""Sets up the training loop.

    The training loop is instantiated from its configuration by using
    the ``BaseTrainingLoop`` factory function.

    Args:
        training_loop (``BaseTrainingLoop`` or dict or None):
            Specifies the training loop or its configuration.
            If ``None``, the ``VanillaTrainingLoop`` is instantiated.

    Returns:
        ``BaseTrainingLoop``: The training loop.
    """
    if training_loop is None:
        training_loop = VanillaTrainingLoop()
    if isinstance(training_loop, dict):
        logger.info(
            "Initializing a training loop from its configuration... "
            f"{str_target_object(training_loop)}"
        )
        training_loop = BaseTrainingLoop.factory(**training_loop)
    return training_loop
