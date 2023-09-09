from __future__ import annotations

__all__ = ["is_loop_observer_config", "setup_loop_observer"]

import logging

from objectory.utils import is_object_config

from gravitorch.loops.observers.base import BaseLoopObserver
from gravitorch.loops.observers.noop import NoOpLoopObserver
from gravitorch.utils.format import str_target_object

logger = logging.getLogger(__name__)


def is_loop_observer_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseLoopObserver``.

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
            for a ``BaseLoopObserver`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.observers import is_loop_observer_config
        >>> is_loop_observer_config({"_target_": "gravitorch.loops.observers.NoOpLoopObserver"})
        True
    """
    return is_object_config(config, BaseLoopObserver)


def setup_loop_observer(loop_observer: BaseLoopObserver | dict | None) -> BaseLoopObserver:
    r"""Sets up a loop observer.

    The loop observer is instantiated from its configuration by
    using the ``BaseLoopObserver`` factory function.

    Args:
    ----
        loop_observer (``BaseLoopObserver`` or dict or None):
            Specifies the loop observer or its configuration.
            If ``None``, the ``NoOpLoopObserver`` is instantiated.

    Returns:
    -------
        ``BaseLoopObserver``: The loop observer.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.observers import setup_loop_observer
        >>> observer = setup_loop_observer(
        ...     {"_target_": "gravitorch.loops.observers.NoOpLoopObserver"}
        ... )
        >>> observer
        NoOpLoopObserver()
    """
    if loop_observer is None:
        loop_observer = NoOpLoopObserver()
    if isinstance(loop_observer, dict):
        logger.info(
            "Initializing a loop observer module from its configuration... "
            f"{str_target_object(loop_observer)}"
        )
        loop_observer = BaseLoopObserver.factory(**loop_observer)
    if not isinstance(loop_observer, BaseLoopObserver):
        logger.warning(
            f"loop_observer is not a `BaseLoopObserver` (received: {type(loop_observer)})"
        )
    return loop_observer
