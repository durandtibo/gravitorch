__all__ = ["log_cuda_memory_summary", "log_max_cuda_memory_allocated"]

import logging

import torch

from gravitorch.utils.format import human_byte_size

logger = logging.getLogger(__name__)


def log_cuda_memory_summary() -> None:
    r"""Logs a CUDA memory summary.

    This function does nothing if CUDA is not available.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.cuda_memory import log_cuda_memory_summary
        >>> torch.randn(4, 6, device='cuda:0')
        tensor([[ 0.9865, -0.5485, -0.9522,  0.0158,  1.1500, -1.9437],
                [ 0.4883,  0.8954, -1.1421,  0.4309, -0.7286,  0.5707],
                [ 0.2984, -1.7045, -0.1694,  0.7324,  0.2014,  0.6356],
                [-0.7928, -1.6999,  0.4724, -0.7914, -0.1426, -0.7764]],
               device='cuda:0')
        >>> log_cuda_memory_summary()
        INFO:gravitorch.utils.cuda_memory:Max CUDA memory allocated: 3.50 KB / 10.92 GB (0.00%)
        INFO:gravitorch.utils.cuda_memory:
        |===========================================================================|
        |                  PyTorch CUDA memory summary, device ID 0                 |
        |---------------------------------------------------------------------------|
        |            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
        |===========================================================================|
        |        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
        |---------------------------------------------------------------------------|
        | Allocated memory      |     512 B  |    3584 B  |   11776 B  |   11264 B  |
        |       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |
        |       from small pool |     512 B  |    3584 B  |   11776 B  |   11264 B  |
        |---------------------------------------------------------------------------|
        | Active memory         |     512 B  |    3584 B  |   11776 B  |   11264 B  |
        |       from large pool |       0 B  |       0 B  |       0 B  |       0 B  |
        |       from small pool |     512 B  |    3584 B  |   11776 B  |   11264 B  |
        |---------------------------------------------------------------------------|
        | GPU reserved memory   |    2048 KB |    2048 KB |    2048 KB |       0 B  |
        |       from large pool |       0 KB |       0 KB |       0 KB |       0 B  |
        |       from small pool |    2048 KB |    2048 KB |    2048 KB |       0 B  |
        |---------------------------------------------------------------------------|
        | Non-releasable memory |    2047 KB |    2047 KB |    2058 KB |   11264 B  |
        |       from large pool |       0 KB |       0 KB |       0 KB |       0 B  |
        |       from small pool |    2047 KB |    2047 KB |    2058 KB |   11264 B  |
        |---------------------------------------------------------------------------|
        | Allocations           |       1    |       7    |      22    |      21    |
        |       from large pool |       0    |       0    |       0    |       0    |
        |       from small pool |       1    |       7    |      22    |      21    |
        |---------------------------------------------------------------------------|
        | Active allocs         |       1    |       7    |      22    |      21    |
        |       from large pool |       0    |       0    |       0    |       0    |
        |       from small pool |       1    |       7    |      22    |      21    |
        |---------------------------------------------------------------------------|
        | GPU reserved segments |       1    |       1    |       1    |       0    |
        |       from large pool |       0    |       0    |       0    |       0    |
        |       from small pool |       1    |       1    |       1    |       0    |
        |---------------------------------------------------------------------------|
        | Non-releasable allocs |       1    |       3    |      10    |       9    |
        |       from large pool |       0    |       0    |       0    |       0    |
        |       from small pool |       1    |       3    |      10    |       9    |
        |---------------------------------------------------------------------------|
        | Oversize allocations  |       0    |       0    |       0    |       0    |
        |---------------------------------------------------------------------------|
        | Oversize GPU segments |       0    |       0    |       0    |       0    |
        |===========================================================================|
    """
    if torch.cuda.is_available():
        log_max_cuda_memory_allocated()
        logger.info(f"\n{torch.cuda.memory_summary()}")


def log_max_cuda_memory_allocated() -> None:
    r"""Logs the max CUDA memory allocated.

    This function does nothing if CUDA is not available.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.utils.cuda_memory import log_max_cuda_memory_allocated
        >>> torch.randn(4, 6, device='cuda:0')
        tensor([[ 0.9865, -0.5485, -0.9522,  0.0158,  1.1500, -1.9437],
                [ 0.4883,  0.8954, -1.1421,  0.4309, -0.7286,  0.5707],
                [ 0.2984, -1.7045, -0.1694,  0.7324,  0.2014,  0.6356],
                [-0.7928, -1.6999,  0.4724, -0.7914, -0.1426, -0.7764]],
               device='cuda:0')
        >>> log_max_cuda_memory_allocated()
        INFO:gravitorch.utils.cuda_memory:Max CUDA memory allocated: 3.50 KB / 10.92 GB (0.00%)
    """
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.mem_get_info()[1]
        logger.info(
            f"Max CUDA memory allocated: {human_byte_size(allocated_memory)} / "
            f"{human_byte_size(total_memory)} "
            f"({100 * allocated_memory / total_memory:.2f}%)"
        )
