from __future__ import annotations

__all__ = ["LogCudaMemory", "LogSysInfo"]

import logging
from types import TracebackType

from gravitorch.rsrc.base import BaseResource
from gravitorch.utils.cudamem import log_max_cuda_memory_allocated
from gravitorch.utils.sysinfo import log_system_info

logger = logging.getLogger(__name__)


class LogCudaMemory(BaseResource):
    r"""Implements a context manager to log the CUDA memory.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import LogCudaMemory
        >>> with LogCudaMemory():
        ...     pass
        ...
    """

    def __enter__(self) -> LogCudaMemory:
        log_max_cuda_memory_allocated()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        log_max_cuda_memory_allocated()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


class LogSysInfo(BaseResource):
    r"""Implements a context manager to log system information.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import LogSysInfo
        >>> with LogSysInfo():
        ...     pass
        ...
    """

    def __enter__(self) -> LogSysInfo:
        log_system_info()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        log_system_info()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"
