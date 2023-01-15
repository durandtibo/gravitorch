r"""This module implements some utilities for the loggers."""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Union


@contextmanager
def disable_logging(level: Union[int, str] = logging.CRITICAL) -> Generator[None, None, None]:
    r"""Context manager to temporarily disable the logging.

    All logging calls of severity ``level`` and below will be
    disabled.

    Args:
        level (int or str): Specifies the level.

    Example usage:

    .. code-block:: python

        >>> import logging
        >>> from gravitorch.utils.logging import disable_logging
        >>> with disable_logging('INFO'):
        ...     logging.critical('CRITICAL')
        ...     logging.info('INFO')
        ...     logging.debug('DEBUG')
        ...
        CRITICAL:root:CRITICAL
    """
    prev_level = logging.getLogger(__name__).level
    if isinstance(level, str):
        level = logging.getLevelName(level)
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(prev_level)
