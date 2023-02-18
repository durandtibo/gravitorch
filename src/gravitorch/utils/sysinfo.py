__all__ = [
    "cpu_human_summary",
    "log_system_info",
    "swap_memory_human_summary",
    "virtual_memory_human_summary",
]

import logging

import psutil

from gravitorch.utils.format import human_byte_size

logger = logging.getLogger(__name__)


def cpu_human_summary() -> str:
    r"""Gets a human-readable summary of the CPU usage.

    Returns:
        str: The human-readable summary

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.sysinfo import cpu_human_summary
        >>> cpu_human_summary()
        CPU - logical/physical count: 4/2 | percent: 42.0 | load 1/5/15min: 42.42/36.48/32.68 %
    """
    loadavg = tuple(100.0 * x / psutil.cpu_count() for x in psutil.getloadavg())
    return (
        f"CPU - logical/physical count: {psutil.cpu_count()}/{psutil.cpu_count(logical=False)} | "
        f"percent: {psutil.cpu_percent()} % | "
        f"load 1/5/15min: {loadavg[0]:.2f}/{loadavg[1]:.2f}/{loadavg[2]:.2f} %"
    )


def log_system_info() -> None:
    r"""Log information about the system.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.sysinfo import log_system_info
        >>> log_system_info()
    """
    logger.info(cpu_human_summary())
    logger.info(virtual_memory_human_summary())
    logger.info(swap_memory_human_summary())


def swap_memory_human_summary() -> str:
    r"""Gets a human-readable summary of the swap memory usage.

    Returns:
        str: The human-readable summary

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.sysinfo import swap_memory_human_summary
        >>> swap_memory_human_summary()
        swap memory - total: 17.00 GB | used: 15.66 GB | free: 1.34 GB | percent: 92.1% | sin: 835.39 GB | sout: 45.64 GB  # noqa: E501,B950
    """
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

    Returns:
        str: The human-readable summary

    Example usage:

    .. code-block:: python

        >>> from gravitorch.utils.sysinfo import virtual_memory_human_summary
        >>> virtual_memory_human_summary()
        virtual memory - total: 16.00 GB | available: 2.89 GB | percent: 81.9% | used: 5.43 GB | free: 28.14 MB  # noqa: E501,B950
    """
    vm = psutil.virtual_memory()
    return (
        f"virtual memory - total: {human_byte_size(vm.total)} | "
        f"available: {human_byte_size(vm.available)} | "
        f"percent: {vm.percent} % | "
        f"used: {human_byte_size(vm.used)} | "
        f"free: {human_byte_size(vm.free)}"
    )
