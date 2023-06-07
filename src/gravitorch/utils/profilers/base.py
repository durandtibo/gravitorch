from __future__ import annotations

__all__ = ["BaseProfiler"]

from abc import ABC, abstractmethod
from types import TracebackType

from objectory import AbstractFactory


class BaseProfiler(ABC, metaclass=AbstractFactory):
    r"""Defines the base profiler class.

    Each profiler should be a context manager.

    .. note::

    The profiler is an experimental feature and may change in the
    future without warnings.
    """

    def __enter__(self) -> BaseProfiler:
        r"""Starts profiling."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        r"""Ends profiling."""

    @abstractmethod
    def step(self) -> None:
        r"""Signals the profiler that the next profiling step has
        started."""
