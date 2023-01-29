r"""This module defines some utility functions for distributed experiments."""

__all__ = [
    "conditional_evaluation",
    "has_slurm_distributed_env_vars",
    "has_torch_distributed_env_vars",
    "is_slurm_job",
    "is_distributed_ready",
    "show_all_slurm_env_vars",
    "show_distributed_context_info",
    "show_distributed_env_vars",
    "show_slurm_env_vars",
    "show_torch_distributed_env_vars",
]

import logging
import os
from typing import Callable

from gravitorch.distributed import comm as dist
from gravitorch.distributed._constants import (
    CUDA_VISIBLE_DEVICES,
    SLURM_DISTRIBUTED_ENV_VARS,
    SLURM_JOB_ID,
    SLURM_NTASKS,
    TORCH_DISTRIBUTED_ENV_VARS,
)
from gravitorch.utils.format import to_pretty_dict_str

logger = logging.getLogger(__name__)


def is_slurm_job() -> bool:
    r"""Indicates if the current process is connected to a SLURM job.

    A process is considered connected to a SLURM job if the
    environment variable ``SLURM_JOB_ID`` exists.

    Returns:
        bool: ``True`` if the current process is connected to a SLURM
            job, otherwise ``False``.
    """
    return SLURM_JOB_ID in os.environ


def is_distributed_ready() -> bool:
    r"""Indicates if the distributed context is ready to be activated or not.

    Returns:
        bool: ``True`` if the distributed context should be activated,
            otherwise ``False``.
    """
    if is_slurm_job():
        return has_slurm_distributed_env_vars() and int(os.environ.get(SLURM_NTASKS, 0)) > 1
    return has_torch_distributed_env_vars()


def has_torch_distributed_env_vars() -> bool:
    r"""Indicates if the environment variables for the native PyTorch
    distributed backends are set or not.

    It assumes that the distributed processes were launched with
    ``torch.distributed.run``
    (https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py)
    or any other tools that setup the same environment variables.

    Returns:
        bool: ``True`` if all the environment variables are set,
            otherwise ``False``.
    """
    for env_var in TORCH_DISTRIBUTED_ENV_VARS:
        if env_var not in os.environ:
            return False
    return True


def has_slurm_distributed_env_vars() -> bool:
    r"""Indicates if the environment variables required to initialize the native
    PyTorch distributed backends are set or not.

    Returns:
        bool: ``True`` if all the environment variables are set,
            otherwise ``False``.
    """
    for env_var in SLURM_DISTRIBUTED_ENV_VARS:
        if env_var not in os.environ:
            return False
    return True


def show_distributed_env_vars() -> None:
    r"""Shows the value of some environment variables used to set up the
    distributed context."""
    show_torch_distributed_env_vars()
    if is_slurm_job():
        show_slurm_env_vars()
        show_all_slurm_env_vars()


def show_torch_distributed_env_vars() -> None:
    r"""Shows the value of the environment variables used to set up the PyTorch
    distributed context.

    This function shows the value of the following environment
    variables:

        - WORLD_SIZE: The total number of processes, so that the
            master knows how many workers to wait for.
        - RANK: The global rank of the process. Rank is a unique
            identifier assigned to each process within a distributed
            process group. They are always consecutive integers
            ranging from 0 to ``WORLD_SIZE``.
        - LOCAL_RANK: The local rank of the process. This value is
            not unique globally, but it is unique per node.
        - MASTER_ADDR: IP address (or hostname) of the machine that
            will host the process with rank 0.
        - MASTER_PORT: The master node (rank 0)'s free port that
            needs to be used for communication during distributed
            training.
        - The remaining environment variables can be found in the
            documentation of ``torch.distributed.run``
            https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py
    """
    names = TORCH_DISTRIBUTED_ENV_VARS + (CUDA_VISIBLE_DEVICES,)
    env_vars = {env_var: os.environ.get(env_var, "<NOT SET>") for env_var in names}
    logger.info(
        "PyTorch environment variables for distributed training:\n"
        f"{to_pretty_dict_str(env_vars, sorted_keys=True, indent=2)}\n"
    )


def show_all_slurm_env_vars() -> None:
    r"""Shows the value of the all the SLURM environment variables."""
    env_vars = {key: value for key, value in os.environ.items() if key.startswith("SLURM_")}
    logger.info(
        "All SLURM environment variables:\n"
        f"{to_pretty_dict_str(env_vars, sorted_keys=True, indent=2)}\n"
    )


def show_slurm_env_vars() -> None:
    r"""Shows the value of some SLURM environment variables."""
    env_vars = {
        env_var: os.environ.get(env_var, "<NOT SET>") for env_var in SLURM_DISTRIBUTED_ENV_VARS
    }
    logger.info(
        "SLURM environment variables:\n"
        f"{to_pretty_dict_str(env_vars, sorted_keys=True, indent=2)}\n"
    )


def show_distributed_context_info() -> None:
    r"""Shows some information about the distributed context.

    This information is useful to verify that each process is well
    configured.
    """
    logger.info(
        f"Config of the distributed context:\n"
        f"  model_name         : {dist.model_name()}\n"
        f"  backend            : {dist.backend()}\n"
        f"  device             : {dist.device()}\n"
        f"  hostname           : {dist.hostname()}\n"
        f"  get_world_size     : {dist.get_world_size()}\n"
        f"  get_rank           : {dist.get_rank()}\n"
        f"  get_local_rank     : {dist.get_local_rank()}\n"
        f"  is_main_process    : {dist.is_main_process()}\n"
        f"  get_nnodes         : {dist.get_nnodes()}\n"
        f"  get_nproc_per_node : {dist.get_nproc_per_node()}\n"
        f"  get_node_rank      : {dist.get_node_rank()}\n"
    )


def conditional_evaluation(only_main_process: bool, callable: Callable, *args, **kwargs) -> None:
    r"""Evaluates or not the callable based on the distributed context.

    Args:
        only_main_process (bool, optional): If ``True``, only the
            main process evaluates the callable otherwise
            all the processes evaluate the callable.
        callable (Callable): Specifies the callable to evaluate.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Example:

    .. code-block:: python

        >>> from gravitorch.distributed.utils import conditional_evaluation
        >>> conditional_evaluation(False, print, "hello")
        hello
        # If main process
        >>> conditional_evaluation(True, print, "hello")
        hello
        # If non main process
        >>> conditional_evaluation(True, print, "hello")
    """
    if only_main_process and not dist.is_main_process():
        return
    callable(*args, **kwargs)
