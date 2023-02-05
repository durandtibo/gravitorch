__all__ = ["NoOpProfiler"]

import logging
from types import TracebackType
from typing import Optional

from gravitorch.utils.profilers.base import BaseProfiler

logger = logging.getLogger(__name__)


class NoOpProfiler(BaseProfiler):
    r"""Implements a no-op profiler.

    This class allows to use a profiler without changing the
    implementation.
    """

    def __enter__(self) -> "NoOpProfiler":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def step(self) -> None:
        r"""Signals the profiler that the next profiling step has started.

        Because it is a no-op class, this method does nothing.
        """
