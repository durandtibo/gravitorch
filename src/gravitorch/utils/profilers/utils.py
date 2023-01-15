r"""This module implements some utility functions for the profilers."""

__all__ = ["setup_profiler"]

import logging
from typing import Union

from gravitorch.utils.format import str_target_object
from gravitorch.utils.profilers import BaseProfiler, NoOpProfiler

logger = logging.getLogger(__name__)


def setup_profiler(profiler: Union[BaseProfiler, dict, None]) -> BaseProfiler:
    r"""Sets up the profiler.

    The profiler is instantiated from its configuration by using the
    ``BaseProfiler`` factory function.

    Args:
        profiler (``BaseProfiler`` or dict or None): Specifies the
            profiler or its configuration. If ``None``, the
            ``NoOpProfiler`` is instantiated.

    Returns:
        ``BaseProfiler``: A profiler.
    """
    if profiler is None:
        profiler = NoOpProfiler()
    if isinstance(profiler, dict):
        logger.info(
            f"Initializing a profiler from its configuration... {str_target_object(profiler)}"
        )
        profiler = BaseProfiler.factory(**profiler)
    return profiler
