r"""This module defines some primitives for distributed communication."""

__all__ = [
    "Backend",
    "UnknownBackendError",
    "all_gather",
    "all_reduce",
    "auto_backend",
    "available_backends",
    "backend",
    "barrier",
    "broadcast",
    "device",
    "distributed_context",
    "finalize",
    "get_local_rank",
    "get_nnodes",
    "get_node_rank",
    "get_nproc_per_node",
    "get_rank",
    "get_world_size",
    "gloocontext",
    "hostname",
    "initialize",
    "is_distributed",
    "is_main_process",
    "model_name",
    "ncclcontext",
    "resolve_backend",
    "set_local_rank",
    "show_config",
]

import logging
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch
from ignite.distributed import utils

from gravitorch.distributed.utils import is_distributed_ready, show_distributed_env_vars

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


class UnknownBackendError(Exception):
    r"""This exception is raised when you try to use an unknown backend."""


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
def distributed_context(backend: str) -> Generator[None, None, None]:
    r"""Context manager to initialize the distributed context for a given
    backend.

    Args:
        backend (str): Specifies the distributed backend to use.
            You can find more information on the distributed backends
            at https://pytorch.org/docs/stable/distributed.html#backends

    Example usage

    .. code-block:: python

        >>> import torch
        >>> from gravitorch import distributed as dist
        >>> with dist.distributed_context(backend='gloo'):
        ...     print(dist.backend())
        ...     x = torch.ones(2, 3, device=dist.device())
        ...     dist.all_reduce(x, op="SUM")
        ...     print(x)
        gloo
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
    """
    show_distributed_env_vars()
    if backend not in available_backends():
        raise UnknownBackendError(
            f"Unknown backend '{backend}'. Available backends: {available_backends()}"
        )

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


def auto_backend() -> Optional[str]:
    r"""Finds the best distributed backend for the current environment.

    The rules to find the best distributed backend are:

        - If the NCCL backend and a GPU are available, the best
            distributed backend is NCCL
        - If the GLOO backend is available, the best distributed
            backend is GLOO
        - Otherwise, ``None`` is returned because there is no
            best distributed backend

    Returns:
        str or ``None``: The name of the best distributed backend.

    Example usage:

    .. code-block:: python

        >>> from gravitorch import distributed as dist
        >>> dist.auto_backend()
        'gloo'
    """
    if torch.cuda.is_available() and Backend.NCCL in available_backends():
        return Backend.NCCL
    if Backend.GLOO in available_backends():
        return Backend.GLOO
    return None


def resolve_backend(backend: Optional[str]) -> Optional[str]:
    r"""Resolves the distributed backend if ``'auto'``.

    Args:
        backend (str or ``None``): Specifies the distributed
            backend. If ``'auto'``, this function will find the best
            option for the distributed backend according to the
            context and some rules.

    Returns:
        str or ``None``: The distributed backend or ``None`` if it
            should not use a distributed backend.

    Example usage:

    .. code-block:: python

        >>> from gravitorch import distributed as dist
        >>> dist.resolve_backend("auto")
        gloo
    """
    if backend is None:
        return None
    if backend == "auto":
        if is_distributed_ready():
            return auto_backend()
        # Set to ``None`` because the process does not seem ready
        # to be configured for a distributed experiment.
        return None
    if backend not in available_backends():
        raise UnknownBackendError(
            f"Unknown distributed backend '{backend}'. Available backends: {available_backends()}"
        )
    return backend


@contextmanager
def gloocontext() -> Generator[None, None, None]:
    r"""Context manager to initialize a GLOO distributed context.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.distributed import backend, gloocontext
        >>> with gloocontext():
        ...     print(backend())
        gloo
    """
    if Backend.GLOO not in available_backends():
        raise RuntimeError(
            f"GLOO backend is not available. Available backends: {available_backends()}"
        )
    with distributed_context(Backend.GLOO):
        yield


@contextmanager
def ncclcontext() -> Generator[None, None, None]:
    r"""Context manager to initialize the NCCL distributed context.

    Raises:
        RuntimeError if CUDA is not available

    Example usage:

    .. code-block:: python

        >>> from gravitorch.distributed import backend, ncclcontext
        >>> with ncclcontext():
        ...     print(backend())
        nccl
    """
    if Backend.NCCL not in available_backends():
        raise RuntimeError(
            f"NCCL backend is not available. Available backends: {available_backends()}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA capable devices but CUDA is not available")
    with distributed_context(Backend.NCCL):
        with torch.cuda.device(get_local_rank()):
            yield


BACKEND_TO_CONTEXT = {
    Backend.NCCL: ncclcontext(),
    Backend.GLOO: gloocontext(),
    None: nullcontext(),
}
