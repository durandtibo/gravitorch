from __future__ import annotations

__all__ = ["NoOpProfiler"]

import logging
from types import TracebackType

from gravitorch.utils.profilers.base import BaseProfiler

logger = logging.getLogger(__name__)


class NoOpProfiler(BaseProfiler):
    r"""Implements a no-op profiler.

    This class allows to use a profiler without changing the
    implementation.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.profilers import NoOpProfiler
        >>> with NoOpProfiler() as profiler:
        ...     x = torch.ones(2, 3)
        ...     for _ in range(20):
        ...         x += x
        ...         profiler.step()
        ...
    """

    def __enter__(self) -> NoOpProfiler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def step(self) -> None:
        r"""Signals the profiler that the next profiling step has
        started.

        Because it is a no-op class, this method does nothing.
        """
