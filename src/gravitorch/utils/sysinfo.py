__all__ = ["cpu_human_summary", "swap_memory_human_summary", "virtual_memory_human_summary"]

from gravitorch.utils.format import to_human_readable_byte_size
from gravitorch.utils.integrations import is_psutil_available

if is_psutil_available():
    import psutil
else:
    psutil = None  # pragma: no cover


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
        f"percent: {psutil.cpu_percent()} | "
        f"load 1/5/15min: {loadavg[0]:.2f}/{loadavg[1]:.2f}/{loadavg[2]:.2f} %"
    )


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
        f"swap memory - total: {to_human_readable_byte_size(swap.total)} | "
        f"used: {to_human_readable_byte_size(swap.used)} | "
        f"free: {to_human_readable_byte_size(swap.free)} | "
        f"percent: {swap.percent}% | "
        f"sin: {to_human_readable_byte_size(swap.sin)} | "
        f"sout: {to_human_readable_byte_size(swap.sout)}"
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
    if psutil is None:
        return "psutil is required to collect virtual memory usage"
    vm = psutil.virtual_memory()
    return (
        f"virtual memory - total: {to_human_readable_byte_size(vm.total)} | "
        f"available: {to_human_readable_byte_size(vm.available)} | "
        f"percent: {vm.percent}% | "
        f"used: {to_human_readable_byte_size(vm.used)} | "
        f"free: {to_human_readable_byte_size(vm.free)}"
    )
