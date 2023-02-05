__all__ = [
    "PyTorchConfig",
    "PyTorchCudaBackend",
    "PyTorchCudaBackendState",
    "PyTorchCudnnBackend",
    "PyTorchCudnnBackendState",
]

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Optional

import torch
from torch.backends import cuda, cudnn

from gravitorch.rsrc.base import BaseResource
from gravitorch.utils.format import to_pretty_dict_str

logger = logging.getLogger(__name__)


class PyTorchConfig(BaseResource):
    r"""Implements a context manager to show the PyTorch configuration."""

    def __enter__(self) -> "PyTorchConfig":
        logger.info(f"PyTorch version: {torch.version.__version__}  ({torch.version.git_version})")
        logger.info(f"PyTorch configuration:\n{torch.__config__.show()}")
        logger.info(f"PyTorch parallel information:\n{torch.__config__.parallel_info()}")
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cap = torch.cuda.get_device_capability(device)
            logger.info(f"PyTorch CUDA compute capability: {'.'.join(str(ver) for ver in cap)}")
            logger.info(f"PyTorch GPU name: {torch.cuda.get_device_name(device)}")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"


@dataclass
class PyTorchCudaBackendState:
    allow_tf32: bool
    allow_fp16_reduced_precision_reduction: bool
    flash_sdp_enabled: bool
    math_sdp_enabled: bool
    preferred_linalg_backend: Any

    def restore(self) -> None:
        r"""Restores the PyTorch CUDA backend configuration by using the values
        in the state."""
        cuda.matmul.allow_tf32 = self.allow_tf32
        cuda.matmul.allow_fp16_reduced_precision_reduction = (
            self.allow_fp16_reduced_precision_reduction
        )
        cuda.enable_math_sdp(self.math_sdp_enabled)
        cuda.enable_flash_sdp(self.flash_sdp_enabled)
        cuda.preferred_linalg_library(self.preferred_linalg_backend)

    @classmethod
    def create(cls) -> "PyTorchCudaBackendState":
        r"""Creates a state to capture the current PyTorch CUDA backend.

        Returns:
            ``PyTorchCudaBackendState``: The current state.
        """
        return cls(
            allow_tf32=cuda.matmul.allow_tf32,
            allow_fp16_reduced_precision_reduction=(
                cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            math_sdp_enabled=cuda.math_sdp_enabled(),
            flash_sdp_enabled=cuda.flash_sdp_enabled(),
            preferred_linalg_backend=cuda.preferred_linalg_library(),
        )


class PyTorchCudaBackend(BaseResource):
    r"""Implements a context manager to configure the PyTorch CUDA backend.

    Args:
        allow_tf32 (bool or ``None``, optional): Specifies the value
            of ``torch.backends.cuda.matmul.allow_tf32``.
            If ``None``, the default value is used. Default: ``None``
        allow_fp16_reduced_precision_reduction (bool or ``None``,
            optional): Specifies the value of
            ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction``.
            If ``None``, the default value is used. Default: ``None``
        flash_sdp_enabled (bool or ``None``, optional): Specifies the
            value  of ``torch.backends.cuda.flash_sdp_enabled``.
            If ``None``, the default value is used. Default: ``None``
        math_sdp_enabled (bool or ``None``, optional): Specifies the
            value of ``torch.backends.cuda.math_sdp_enabled``.
            If ``None``, the default value is used. Default: ``None``
        preferred_linalg_backend (str or ``None``, optional):
            Specifies the value of
            ``torch.backends.cuda.preferred_linalg_library``.
            If ``None``, the default value is used. Default: ``None``
        log_info (bool, optional): If ``True``, the state is shown
            after the context manager is created. Default: ``False``
    """

    def __init__(
        self,
        allow_tf32: Optional[bool] = None,
        allow_fp16_reduced_precision_reduction: Optional[bool] = None,
        flash_sdp_enabled: Optional[bool] = None,
        math_sdp_enabled: Optional[bool] = None,
        preferred_linalg_backend: Optional[str] = None,
        log_info: bool = False,
    ):
        self._allow_tf32 = allow_tf32
        self._allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        self._flash_sdp_enabled = flash_sdp_enabled
        self._math_sdp_enabled = math_sdp_enabled
        self._preferred_linalg_backend = preferred_linalg_backend

        self._log_info = bool(log_info)
        self._state: list[PyTorchCudaBackendState] = []

    def __enter__(self) -> "PyTorchCudaBackend":
        logger.info("Configuring CUDA backend...")
        self._state.append(PyTorchCudaBackendState.create())
        self._configure()
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logger.info("Restoring CUDA backend configuration...")
        self._state.pop().restore()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(allow_tf32={self._allow_tf32}, "
            "allow_fp16_reduced_precision_reduction="
            f"{self._allow_fp16_reduced_precision_reduction}, "
            f"flash_sdp_enabled={self._flash_sdp_enabled}, "
            f"math_sdp_enabled={self._math_sdp_enabled}, "
            f"preferred_linalg_backend={self._preferred_linalg_backend}, "
            f"log_info={self._log_info})"
        )

    def _configure(self) -> None:
        if self._allow_tf32 is not None:
            cuda.matmul.allow_tf32 = self._allow_tf32
        if self._allow_fp16_reduced_precision_reduction is not None:
            cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self._allow_fp16_reduced_precision_reduction
            )
        if self._flash_sdp_enabled is not None:
            cuda.enable_flash_sdp(self._flash_sdp_enabled)
        if self._math_sdp_enabled is not None:
            cuda.enable_math_sdp(self._math_sdp_enabled)
        if self._preferred_linalg_backend is not None:
            cuda.preferred_linalg_library(self._preferred_linalg_backend)

    def _show(self) -> None:
        prefix = "torch.backends.cuda"
        info = {
            f"{prefix}.matmul.allow_fp16_reduced_precision_reduction": (
                cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            f"{prefix}.matmul.allow_tf32": cuda.matmul.allow_tf32,
            f"{prefix}.is_built": cuda.is_built(),
            f"{prefix}.flash_sdp_enabled": cuda.flash_sdp_enabled(),
            f"{prefix}.math_sdp_enabled": cuda.math_sdp_enabled(),
            f"{prefix}.preferred_linalg_library": cuda.preferred_linalg_library(),
            "torch.version.cuda": torch.version.cuda,
        }
        logger.info(f"CUDA backend:\n{to_pretty_dict_str(info, sorted_keys=True, indent=2)}\n")


@dataclass
class PyTorchCudnnBackendState:
    allow_tf32: bool
    benchmark: bool
    benchmark_limit: Optional[int]
    deterministic: bool
    enabled: bool

    def restore(self) -> None:
        r"""Restores the PyTorch CUDNN backend configuration by using the
        values in the state."""
        cudnn.allow_tf32 = self.allow_tf32
        cudnn.benchmark = self.benchmark
        cudnn.benchmark_limit = self.benchmark_limit
        cudnn.deterministic = self.deterministic
        cudnn.enabled = self.enabled

    @classmethod
    def create(cls) -> "PyTorchCudnnBackendState":
        r"""Creates a state to capture the current PyTorch CUDA CUDNN.

        Returns:
            ``PyTorchCudnnBackendState``: The current state.
        """
        return cls(
            allow_tf32=cudnn.allow_tf32,
            benchmark=cudnn.benchmark,
            benchmark_limit=cudnn.benchmark_limit,
            deterministic=cudnn.deterministic,
            enabled=cudnn.enabled,
        )


class PyTorchCudnnBackend(BaseResource):
    r"""Implements a context manager to configure the PyTorch CUDNN backend.

    Args:
        allow_tf32 (bool or ``None``, optional): Specifies the value
            of ``torch.backends.cudnn.allow_tf32``. If ``None``,
            the default value is used. Default: ``None``
        benchmark (bool or ``None``, optional): Specifies the value of
            ``torch.backends.cudnn.benchmark``. If ``None``,
            the default value is used. Default: ``None``
        benchmark_limit (int or ``None``, optional): Specifies the
            value of ``torch.backends.cudnn.benchmark_limit``.
            If ``None``, the default value is used. Default: ``None``
        deterministic (bool or ``None``, optional): Specifies the
            value of ``torch.backends.cudnn.deterministic``.
            If ``None``, the default value is used. Default: ``None``
        enabled (bool or ``None``, optional): Specifies the value of
            ``torch.backends.cudnn.enabled``. If ``None``,
            the default value is used. Default: ``None``
        log_info (bool, optional): If ``True``, the state is shown
            after the context manager is created. Default: ``False``
    """

    def __init__(
        self,
        allow_tf32: bool = None,
        benchmark: Optional[bool] = None,
        benchmark_limit: Optional[int] = None,
        deterministic: Optional[bool] = None,
        enabled: Optional[bool] = None,
        log_info: bool = False,
    ):
        self._allow_tf32 = allow_tf32
        self._benchmark = benchmark
        self._benchmark_limit = benchmark_limit
        self._deterministic = deterministic
        self._enabled = enabled

        self._log_info = bool(log_info)
        self._state: list[PyTorchCudnnBackendState] = []

    def __enter__(self) -> "PyTorchCudnnBackend":
        logger.info("Configuring CUDNN backend...")
        self._state.append(PyTorchCudnnBackendState.create())
        self._configure()
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logger.info("Restoring CUDNN backend configuration...")
        self._state.pop().restore()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(allow_tf32={self._allow_tf32}, "
            f"benchmark={self._benchmark}, benchmark_limit={self._benchmark_limit}, "
            f"deterministic={self._deterministic}, enabled={self._enabled}, "
            f"log_info={self._log_info})"
        )

    def _configure(self) -> None:
        if self._allow_tf32 is not None:
            cudnn.allow_tf32 = self._allow_tf32
        if self._benchmark is not None:
            cudnn.benchmark = self._benchmark
        if self._benchmark_limit is not None:
            cudnn.benchmark_limit = self._benchmark_limit
        if self._deterministic is not None:
            cudnn.deterministic = self._deterministic
        if self._enabled is not None:
            cudnn.enabled = self._enabled

    def _show(self) -> None:
        prefix = "torch.backends.cudnn"
        info = {
            f"{prefix}.allow_tf32": cudnn.allow_tf32,
            f"{prefix}.benchmark": cudnn.benchmark,
            f"{prefix}.benchmark_limit": cudnn.benchmark_limit,
            f"{prefix}.deterministic": cudnn.deterministic,
            f"{prefix}.enabled": cudnn.enabled,
            f"{prefix}.is_available": cudnn.is_available(),
            f"{prefix}.version": cudnn.version(),
        }
        logger.info(f"CUDNN backend:\n{to_pretty_dict_str(info, sorted_keys=True, indent=2)}\n")
