r"""This module implements a PyTorch profiler."""

__all__ = ["PyTorchProfiler"]

import logging
from types import TracebackType
from typing import Optional

import torch

from gravitorch.utils.profilers.base import BaseProfiler

logger = logging.getLogger(__name__)


class PyTorchProfiler(BaseProfiler):
    r"""Implements a PyTorch profiler.

    This profiler relies on the package ``torch.profiler`` to profile
    the code.

    Args:
        profiler (``torch.profiler.profile``): Specifies the profiler
            to use.
    """

    def __init__(self, profiler: torch.profiler.profile):
        self._profiler = profiler

    def __enter__(self) -> "PyTorchProfiler":
        logger.info("Starting PyTorch profiler...")
        self._profiler.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logger.info("Ending PyTorch profiler")
        self._profiler.__exit__(exc_type, exc_val, exc_tb)

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def step(self) -> None:
        r"""Signals the profiler that the next profiling step has started."""
        self._profiler.step()

    @classmethod
    def scheduled_profiler_with_tensorboard_trace(
        cls,
        trace_path: str,
        wait: int,
        warmup: int,
        active: int,
        repeat: int = 0,
        skip_first: int = 0,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
    ) -> "PyTorchProfiler":
        r"""Implements a scheduled profiler with a TensorBoard trace.

        The profiler will skip the first ``skip_first`` steps, then
        wait for ``wait`` steps, then do the warmup for the next
        ``warmup`` steps, then do the active recording for the next
        ``active`` steps and then repeat the cycle starting with
        ``wait`` steps. The optional number of cycles is specified
        with the ``repeat`` parameter, the zero value means that the
        cycles will continue until the profiling is finished.

        Args:
            trace_path (str): Specifies the path where to write the
                profiling trace. This path can be directly delivered
                to TensorBoard as logdir.
            wait (int): Specifies the number of waiting steps in a
                cycle.
            warmup (int): Specifies the number of warmup steps in a
                cycle.
            active (int): Specifies the number of active steps in a
                cycle.
            repeat (int, optional): Specifies the number of cycles.
                If ``0``, the cycles will continue until the profiling
                is finished. Default: ``0``
            skip_first (int, optional): Specifies the number of steps
                that are skipped at the beginning. Default: ``0``
            record_shapes (bool, optional): If ``True``, the profiler
                saves information about operator’s input shapes.
                Default: ``False``
            profile_memory (bool, optional): If ``True``, the profiler
                tracks tensor memory allocation/deallocation.
                Default: ``False``
            with_stack (bool, optional): If ``True``, the profiler
                record source information (file and line number) for
                the ops. Default: ``False``
            with_flops (bool, optional): If ``True``, the profiler
                uses formula to estimate the FLOPS of specific
                operators (matrix multiplication and 2D convolution).
                Default: ``False``

        Returns:
            ``PyTorchProfiler``: A scheduled profiler with a TensorBoard trace.
        """
        return cls(
            torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=wait,
                    warmup=warmup,
                    active=active,
                    repeat=repeat,
                    skip_first=skip_first,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
            )
        )
