from __future__ import annotations

__all__ = [
    "cpu_human_summary",
    "log_system_info",
    "swap_memory_human_summary",
    "virtual_memory_human_summary",
]

import logging

from gravitorch.utils.format import human_byte_size
from gravitorch.utils.imports import is_psutil_available

if is_psutil_available():  # pragma: no cover
    import psutil

logger = logging.getLogger(__name__)


def cpu_human_summary() -> str:
    r"""Gets a human-readable summary of the CPU usage.

    Returns
    -------
        str: The human-readable summary

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.sysinfo import cpu_human_summary
        >>> cpu_human_summary()
    """
    if not is_psutil_available():
        logger.warning(
            "`psutil` package is required to compute the CPU usage summary. "
            "You can install `psutil` package with the command:\n\n"
            "pip install psutil\n"
        )
        return "CPU - N/A"
    loadavg = tuple(100.0 * x / psutil.cpu_count() for x in psutil.getloadavg())
    return (
        f"CPU - logical/physical count: {psutil.cpu_count()}/{psutil.cpu_count(logical=False)} | "
        f"percent: {psutil.cpu_percent()} % | "
        f"load 1/5/15min: {loadavg[0]:.2f}/{loadavg[1]:.2f}/{loadavg[2]:.2f} %"
    )


def log_system_info() -> None:
    r"""Log information about the system.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.sysinfo import log_system_info
        >>> log_system_info()
    """
    logger.info(cpu_human_summary())
    logger.info(virtual_memory_human_summary())
    logger.info(swap_memory_human_summary())


def swap_memory_human_summary() -> str:
    r"""Gets a human-readable summary of the swap memory usage.

    Returns
    -------
        str: The human-readable summary

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.sysinfo import swap_memory_human_summary
        >>> swap_memory_human_summary()
    """
    if not is_psutil_available():
        logger.warning(
            "`psutil` package is required to compute the swap memory usage summary. "
            "You can install `psutil` package with the command:\n\n"
            "pip install psutil\n"
        )
        return "swap memory - N/A"
    swap = psutil.swap_memory()
    return (
        f"swap memory - total: {human_byte_size(swap.total)} | "
        f"used: {human_byte_size(swap.used)} | "
        f"free: {human_byte_size(swap.free)} | "
        f"percent: {swap.percent} % | "
        f"sin: {human_byte_size(swap.sin)} | "
        f"sout: {human_byte_size(swap.sout)}"
    )


def virtual_memory_human_summary() -> str:
    r"""Gets a human-readable summary of the virtual memory usage.

    Returns
    -------
        str: The human-readable summary

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.sysinfo import virtual_memory_human_summary
        >>> virtual_memory_human_summary()
    """
    if not is_psutil_available():
        logger.warning(
            "`psutil` package is required to compute the virtual memory usage summary. "
            "You can install `psutil` package with the command:\n\n"
            "pip install psutil\n"
        )
        return "virtual memory - N/A"
    vm = psutil.virtual_memory()
    return (
        f"virtual memory - total: {human_byte_size(vm.total)} | "
        f"available: {human_byte_size(vm.available)} | "
        f"percent: {vm.percent} % | "
        f"used: {human_byte_size(vm.used)} | "
        f"free: {human_byte_size(vm.free)}"
    )
