r"""This module implements some utility functions to automatically do some
operations.

These implementations cover a large set of use cases but you may need to
implement your own functions for some special use cases.
"""

__all__ = ["auto_distributed_context", "auto_dist_backend", "auto_ddp_model"]

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

import torch
from torch.nn import Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from gravitorch.distributed import comm as dist
from gravitorch.distributed.utils import show_distributed_context_info
from gravitorch.nn.utils.module_helpers import (
    has_learnable_parameters,
    is_module_on_device,
)

logger = logging.getLogger(__name__)


@contextmanager
def auto_distributed_context(dist_backend: Optional[str]) -> Generator[None, None, None]:
    r"""Context manager that automatically initializes and sets up the
    distributed context.

    This context manager provides an implementation to initialize and
    set up the distributed context that covers a large set of use
    cases, but you may need to implement your own functions for some
    special use cases. If the distributed backend is not ``None``,
    this context manager initialize the distributed context. If there
    is at least one GPU available, the selected device is set to the
    local rank. This context manager is a no-op if the distributed
    backend is ``None``.

    Args:
        dist_backend (str or None): Specifies the distributed backend
            to use.
    """
    if dist_backend is not None:  # distributed mode
        with dist.setup_distributed_context(backend=dist_backend):
            if torch.cuda.is_available():  # GPU
                with torch.cuda.device(dist.get_local_rank()):
                    show_distributed_context_info()
                    yield
            else:  # CPU
                show_distributed_context_info()
                yield
    else:  # non-distributed mode
        yield


def auto_dist_backend() -> str:
    r"""Finds the best distributed backend for the current environment.

    The rules to find the best distributed backend are:

        - If there is a GPU is available, the distributed backend is
            NCCL
        - otherwise the backend is GLOO

    Returns:
        str: The name of the backend.
    """
    if torch.cuda.is_available() and dist.Backend.NCCL in dist.available_backends():
        return dist.Backend.NCCL
    return dist.Backend.GLOO


def auto_ddp_model(model: Module, sync_batch_norm: bool = False, **kwargs) -> Module:
    r"""Wraps the model to be ready for Distributed Data Parallel (DDP)
    training.

    Internally, this function does the following steps:

    - Send model to current device if model's parameters are not on
        the device.
    - If NCCL backend, the ``BatchNorm*D`` modules can be converted
        to ``SyncBatchNorm`` modules.
    - Wrap the model to ``DistributedDataParallel`` for native torch
        distributed (NCCL or GLOO) if world size is larger than 1.

    Args:
        model (``torch.nn.Module``): Specifies the model to wrap
            with ``DistributedDataParallel``.
        sync_batch_norm (bool): Specifies if the ``BatchNorm*D``
            modules should be converted to ``SyncBatchNorm`` modules
            (only for NCCL backend).
        **kwargs: Variable arguments for the
            ``DistributedDataParallel`` module. If you use the NCCL
            backend, you cannot set the ``device_ids`` because this
            input is automatically set in this function.

    Returns:
        ``torch.nn.Module``: The model wrapped in a
            ``DistributedDataParallel`` module.
    """
    if isinstance(model, DistributedDataParallel):
        logger.warning(
            "The model is already an instance of ``torch.nn.parallel.DistributedDataParallel`` "
            "so no operation is performed"
        )
        return model
    model = _manage_model_device(model)
    if has_learnable_parameters(model) and dist.get_world_size() > 1:
        model = _wrap_distributed_data_parallel(model, sync_batch_norm, **kwargs)
    return model


def _manage_model_device(model: Module) -> Module:
    r"""Manages the devices of the model.

    Internally, this function moves model's parameters to device
        if its parameters are not on the device.

    Args:
        model (``torch.nn.Module``): Specifies the model to manage
            the devices.
    """
    device = dist.device()
    logger.info(f"Move model parameters to {device}")
    if not is_module_on_device(module=model, device=device):
        model.to(device)
    return model


def _wrap_distributed_data_parallel(
    model: Module, sync_batch_norm: bool = False, **kwargs
) -> Module:
    r"""Wraps the model with the ``DistributedDataParallel`` module.

    Args:
        model (``torch.nn.Module``): Specifies the model to wrap with
            ``DistributedDataParallel``.
        sync_batch_norm (bool): Specifies if the ``BatchNorm*D``
            modules should be converted to ``SyncBatchNorm`` modules
            (only for NCCL backend).
        **kwargs: Variable arguments for the
            ``DistributedDataParallel`` module. You cannot set the
            ``device_ids`` because this input is automatically set
            in this function.

    Returns:
        ``torch.nn.Module``: The model wrapped in a
            ``DistributedDataParallel`` module.
    """
    backend = dist.backend()
    if backend == dist.Backend.NCCL:
        if sync_batch_norm:
            logger.info("Convert batch norm to sync batch norm")
            model = SyncBatchNorm.convert_sync_batchnorm(model)

        lrank = dist.get_local_rank()
        logger.info(
            f"Apply torch DistributedDataParallel on model, device id: {lrank}",
        )
        return DistributedDataParallel(model, device_ids=[lrank], **kwargs)
    if backend == dist.Backend.GLOO:
        logger.info("Apply torch DistributedDataParallel on model")
        return DistributedDataParallel(model, **kwargs)
    return model
