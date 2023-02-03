__all__ = ["Logging", "LoggingState"]

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Union

from gravitorch.distributed import comm as dist
from gravitorch.rsrc.base import BaseResource

logger = logging.getLogger(__name__)


@dataclass
class LoggingState:
    disabled_level: int

    def restore(self) -> None:
        r"""Restores the logging configuration by using the values in the
        state."""
        logging.disable(self.disabled_level)

    @classmethod
    def create(cls) -> "LoggingState":
        r"""Creates a state to capture the current logging configuration.

        Returns:
            ``LoggingState``: The current logging state.
        """
        return cls(disabled_level=logging.root.manager.disable)


class Logging(BaseResource):
    r"""Implements a context manager to disable the logging.

    Args:
        only_main_process (bool, optional): If ``True``, only the
            outputs of the main process are logged. The logging of
            other processes is limited to the error level or above.
            If ``False``, the outputs of all the processes are logged.
            Default: ``True``
        disabled_level (int or str, optional): All logging calls
            of severity ``disabled_level`` and below will be
            disabled. Default: ``39``
        log_info (bool, optional): If ``True``, the state is shown
            after the context manager is created. Default: ``False``
    """

    def __init__(
        self, only_main_process: bool = False, disabled_level: Union[int, str] = logging.ERROR - 1
    ):
        self._only_main_process = bool(only_main_process)
        if isinstance(disabled_level, str):
            disabled_level = logging.getLevelName(disabled_level)
        self._disabled_level = int(disabled_level)

        self._state: list[LoggingState] = []

    def __enter__(self) -> "Logging":
        logger.info("Setting logging configuration...")
        self._state.append(LoggingState.create())
        if self._only_main_process and not dist.is_main_process():
            logging.disable(self._disabled_level)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logger.info("Restoring previous logging configuration...")
        self._state.pop().restore()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(only_main_process={self._only_main_process}, "
            f"disabled_level={self._disabled_level})"
        )
