__all__ = ["setup_loop_observer"]

import logging
from typing import Union

from gravitorch.loops.observers.base import BaseLoopObserver
from gravitorch.loops.observers.noop import NoOpLoopObserver
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def setup_loop_observer(loop_observer: Union[BaseLoopObserver, dict, None]) -> BaseLoopObserver:
    r"""Sets up a loop observer.

    The loop observer is instantiated from its configuration by
    using the ``BaseLoopObserver`` factory function.

    Args:
        loop_observer (``BaseLoopObserver`` or dict or None):
            Specifies the loop observer or its configuration.
            If ``None``, the ``NoOpLoopObserver`` is instantiated.

    Returns:
        ``BaseLoopObserver``: The loop observer.
    """
    if loop_observer is None:
        loop_observer = NoOpLoopObserver()
    if isinstance(loop_observer, dict):
        logger.info(
            "Initializing a loop observer module from its configuration... "
            f"{str_target_object(loop_observer)}"
        )
        loop_observer = BaseLoopObserver.factory(**loop_observer)
    return loop_observer
