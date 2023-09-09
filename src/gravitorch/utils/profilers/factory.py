r"""This module implements some utility functions for the profilers."""

from __future__ import annotations

__all__ = ["is_profiler_config", "setup_profiler"]

import logging

from objectory.utils import is_object_config

from gravitorch.utils.format import str_target_object
from gravitorch.utils.profilers.base import BaseProfiler
from gravitorch.utils.profilers.noop import NoOpProfiler

logger = logging.getLogger(__name__)


def is_profiler_config(config: dict) -> bool:
    r"""Indicates if the input configuration is a configuration for a
    ``BaseProfiler``.

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
            for a ``BaseProfiler`` object.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.profilers import is_profiler_config
        >>> is_profiler_config({"_target_": "gravitorch.utils.profilers.NoOpProfiler"})
        True
    """
    return is_object_config(config, BaseProfiler)


def setup_profiler(profiler: BaseProfiler | dict | None) -> BaseProfiler:
    r"""Sets up the profiler.

    The profiler is instantiated from its configuration by using the
    ``BaseProfiler`` factory function.

    Args:
    ----
        profiler (``BaseProfiler`` or dict or None): Specifies the
            profiler or its configuration. If ``None``, the
            ``NoOpProfiler`` is instantiated.

    Returns:
    -------
        ``BaseProfiler``: A profiler.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.profilers import setup_profiler
        >>> setup_profiler({"_target_": "gravitorch.utils.profilers.NoOpProfiler"})
        NoOpProfiler()
    """
    if profiler is None:
        profiler = NoOpProfiler()
    if isinstance(profiler, dict):
        logger.info(
            f"Initializing a profiler from its configuration... {str_target_object(profiler)}"
        )
        profiler = BaseProfiler.factory(**profiler)
    return profiler
