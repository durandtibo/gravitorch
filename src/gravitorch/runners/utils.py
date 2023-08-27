from __future__ import annotations

__all__ = ["configure_pytorch", "show_cuda_info", "show_cudnn_info", "setup_runner"]

import logging

from torch.backends import cuda, cudnn

from gravitorch.runners.base import BaseRunner
from gravitorch.utils.format import str_pretty_dict, str_target_object

logger = logging.getLogger(__name__)


def setup_runner(runner: BaseRunner | dict) -> BaseRunner:
    r"""Sets up the runner.

    The runner is instantiated from its configuration by using the
    ``BaseRunner`` factory function.

    Args:
    ----
        runner (``BaseRunner`` or dict): Specifies the runner or its
            configuration.

    Returns:
    -------
        ``BaseRunner``: The instantiated runner.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import setup_runner
        >>> runner = setup_runner({"_target_": "gravitorch.runners.TrainingRunner", "engine": {}})
        >>> runner
        TrainingRunner(
          (engine): {}
          (exp_tracker): None
          (handlers): ()
          (resources): ()
          (random_seed): 10139531598155730726
        )
    """
    if isinstance(runner, dict):
        logger.debug(f"Initializing a runner from its configuration... {str_target_object(runner)}")
        runner = BaseRunner.factory(**runner)
    if not isinstance(runner, BaseRunner):
        logger.warning(f"runner is not a BaseRunner (received: {type(runner)})")
    return runner


def configure_pytorch(cudnn_benchmark: bool = False, cudnn_deterministic: bool = False) -> None:
    r"""Configure some PyTorch options.

    Args:
    ----
        cudnn_benchmark (bool, optional): If True, causes cuDNN to
            benchmark multiple convolution algorithms and select the
            fastest. You should not set this option to true if you
            want to reproduce result. See
            https://pytorch.org/docs/stable/notes/randomness.html
            for more information. Default: False
        cudnn_deterministic (bool, optional): If True, causes cuDNN
            to only use deterministic convolution algorithms.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners import configure_pytorch
        >>> configure_pytorch(cudnn_deterministic=True)
    """
    cudnn.benchmark = cudnn_benchmark
    cudnn.deterministic = cudnn_deterministic
    show_cuda_info()
    show_cudnn_info()


def show_cuda_info() -> None:
    r"""Shows information about the CUDA backend.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners.utils import show_cuda_info
        >>> show_cuda_info()
    """
    prefix = "torch.backends.cuda.matmul"
    info = {
        f"{prefix}.allow_fp16_reduced_precision_reduction": (
            cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        f"{prefix}.allow_tf32": cuda.matmul.allow_tf32,
        f"{prefix}.is_built": cuda.is_built(),
    }
    logger.info(f"CUDA backend information:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n")


def show_cudnn_info() -> None:
    r"""Shows information about the cuDNN backend.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.runners.utils import show_cudnn_info
        >>> show_cudnn_info()
    """
    info = {
        "torch.backends.cudnn.allow_tf32": cudnn.allow_tf32,
        "torch.backends.cudnn.benchmark": cudnn.benchmark,
        "torch.backends.cudnn.deterministic": cudnn.deterministic,
        "torch.backends.cudnn.enabled": cudnn.enabled,
        "torch.backends.cudnn.is_available": cudnn.is_available(),
        "torch.backends.cudnn.version": cudnn.version(),
    }
    logger.info(
        f"cuDNN backend information:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n"
    )
