__all__ = ["LogSysInfo"]

import logging
from types import TracebackType
from typing import Optional

from gravitorch.experimental.rsrc.base import BaseResource
from gravitorch.utils.sysinfo import (
    cpu_human_summary,
    swap_memory_human_summary,
    virtual_memory_human_summary,
)

logger = logging.getLogger(__name__)


class LogSysInfo(BaseResource):
    r"""Implements a context manager to log system information."""

    def __enter__(self) -> "LogSysInfo":
        self._show()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._show()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def _show(self) -> None:
        logger.info(cpu_human_summary())
        logger.info(virtual_memory_human_summary())
        logger.info(swap_memory_human_summary())
