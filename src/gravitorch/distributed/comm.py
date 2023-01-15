r"""This module defines some primitives for distributed communication."""

__all__ = [
    "Backend",
    "UnknownBackendError",
    "all_gather",
    "all_reduce",
    "available_backends",
    "backend",
    "barrier",
    "broadcast",
    "device",
    "finalize",
    "get_local_rank",
    "get_nnodes",
    "get_node_rank",
    "get_nproc_per_node",
    "get_rank",
    "get_world_size",
    "hostname",
    "initialize",
    "is_distributed",
    "is_main_process",
    "model_name",
    "set_local_rank",
    "setup_distributed_context",
    "show_config",
]

import logging
from contextlib import contextmanager

from ignite.distributed import utils

from gravitorch.distributed.utils import show_distributed_env_vars

logger = logging.getLogger(__name__)


class Backend:
    r"""Defines the name of the distributed backends currently supported."""
    NCCL = "nccl"
    GLOO = "gloo"


# Do not use ignite directly because it will give more freedom if we want to change one day.
# Only this file should call directly the PyTorch Ignite functions.
all_gather = utils.all_gather
all_reduce = utils.all_reduce
available_backends = utils.available_backends
backend = utils.backend
barrier = utils.barrier
broadcast = utils.broadcast
device = utils.device
finalize = utils.finalize
get_local_rank = utils.get_local_rank
get_nnodes = utils.get_nnodes
get_node_rank = utils.get_node_rank
get_nproc_per_node = utils.get_nproc_per_node
get_rank = utils.get_rank
get_world_size = utils.get_world_size
hostname = utils.hostname
initialize = utils.initialize
model_name = utils.model_name
set_local_rank = utils.set_local_rank
show_config = utils.show_config


def is_main_process() -> bool:
    r"""Indicates if this process is the main process.

    By definition, the main process is the process with the global
    rank 0.

    Returns:
        bool: ``True`` if it is the main process, otherwise ``False``.
    """
    return get_rank() == 0


def is_distributed():
    r"""Indicates if the current process is part of a distributed group.

    Returns:
        bool: ``True`` if the current process is part of a distributed
            group, otherwise ``False``.
    """
    return get_world_size() > 1


@contextmanager
def setup_distributed_context(backend: str) -> None:
    r"""Context manager to set up the distributed context for a given backend.

    Args:
        backend (str): Specifies the distributed backend to use.
            You can find more information on the distributed backends
            at https://pytorch.org/docs/stable/distributed.html#backends

    Example usage

    .. code-block:: python

        >>> import torch
        >>> from gravitorch import distributed as dist
        >>> with dist.setup_distributed_context(backend='gloo'):
        ...     print(dist.backend())
        ...     x = torch.ones(2, 3, device=dist.device())
        ...     dist.all_reduce(x, op="SUM")
        ...     print(x)
        ... # The distributed backend is deactivated.
        gloo
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
    """
    show_distributed_env_vars()
    logger.info(f"Available distributed backends: {available_backends()}")
    if backend not in available_backends():
        raise UnknownBackendError(
            f"Unknown backend '{backend}'. Available backends: {available_backends()}"
        )

    # Initialize the distributed context.
    initialize(backend, init_method="env://")

    try:
        # Distributed processes synchronization is needed here to
        # prevent a possible timeout after calling init_process_group.
        # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
        barrier()
        yield
    finally:
        logger.info("Destroying the distributed process...")
        finalize()
        logger.info("Distributed process destroyed")


class UnknownBackendError(Exception):
    r"""This exception is raised when you try to use an unknown backend."""
