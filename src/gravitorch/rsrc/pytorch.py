from __future__ import annotations

__all__ = [
    "PyTorchConfig",
    "PyTorchConfigState",
    "PyTorchCudaBackend",
    "PyTorchCudaBackendState",
    "PyTorchCudnnBackend",
    "PyTorchCudnnBackendState",
    "PyTorchMpsBackend",
    "PyTorchMpsBackendState",
]

import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch.backends import cuda, cudnn, mps

from gravitorch.rsrc.base import BaseResource
from gravitorch.utils.format import str_pretty_dict

logger = logging.getLogger(__name__)


@dataclass
class PyTorchConfigState:
    float32_matmul_precision: str
    deterministic_algorithms_mode: bool
    deterministic_algorithms_warn_only: bool

    def restore(self) -> None:
        r"""Restores the PyTorch configuration by using the values in
        the state."""
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch.use_deterministic_algorithms(
            mode=self.deterministic_algorithms_mode,
            warn_only=self.deterministic_algorithms_warn_only,
        )

    @classmethod
    def create(cls) -> PyTorchConfigState:
        r"""Creates a state to capture the current PyTorch configuration.

        Returns
        -------
            ``PyTorchConfigState``: The current state.
        """
        return cls(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            deterministic_algorithms_mode=torch.are_deterministic_algorithms_enabled(),
            deterministic_algorithms_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )


class PyTorchConfig(BaseResource):
    r"""Implements a context manager to show the PyTorch configuration.

    Args:
        float32_matmul_precision (``str`` or ``None``, optional):
            Specifies the internal precision of float32 matrix
            multiplications. Default: ``None``
        deterministic_algorithms_mode (``bool`` or ``None``,
            optional): If True, makes potentially nondeterministic
            operations switch to a deterministic algorithm or throw
            a runtime error. If False, allows nondeterministic
            operations. Default: ``None``
        deterministic_algorithms_warn_only (``bool``, optional):
            If True, operations that do not have a deterministic
            implementation will throw a warning instead of an error.
            Default: ``False``
        log_info (``bool``, optional): If ``True``, information about
            the resource is logged after the resource is initialized.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import PyTorchConfig
        >>> with PyTorchConfig(float32_matmul_precision="highest"):
        ...     pass
        ...
    """

    def __init__(
        self,
        float32_matmul_precision: str | None = None,
        deterministic_algorithms_mode: bool | None = None,
        deterministic_algorithms_warn_only: bool = False,
        log_info: bool = False,
    ) -> None:
        self._float32_matmul_precision = float32_matmul_precision
        self._deterministic_algorithms_mode = deterministic_algorithms_mode
        self._deterministic_algorithms_warn_only = deterministic_algorithms_warn_only

        self._log_info = bool(log_info)
        self._state: list[PyTorchConfigState] = []

    def __enter__(self) -> PyTorchConfig:
        logger.info(f"PyTorch version: {torch.version.__version__}  ({torch.version.git_version})")
        logger.info(f"PyTorch configuration:\n{torch.__config__.show()}")
        logger.info(f"PyTorch parallel information:\n{torch.__config__.parallel_info()}")
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cap = torch.cuda.get_device_capability(device)
            logger.info(f"PyTorch CUDA compute capability: {'.'.join(str(ver) for ver in cap)}")
            logger.info(f"PyTorch GPU name: {torch.cuda.get_device_name(device)}")

        logger.info("Configuring PyTorch...")
        self._state.append(PyTorchConfigState.create())
        self._configure()
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("Restoring PyTorch configuration...")
        self._state.pop().restore()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"float32_matmul_precision={self._float32_matmul_precision}, "
            f"deterministic_algorithms_mode={self._deterministic_algorithms_mode}, "
            f"deterministic_algorithms_warn_only={self._deterministic_algorithms_warn_only})"
        )

    def _configure(self) -> None:
        if self._float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self._float32_matmul_precision)
        if self._deterministic_algorithms_mode is not None:
            torch.use_deterministic_algorithms(
                mode=self._deterministic_algorithms_mode,
                warn_only=self._deterministic_algorithms_warn_only,
            )

    def _show(self) -> None:
        prefix = "torch"
        info = {
            f"{prefix}.float32_matmul_precision": torch.get_float32_matmul_precision(),
            f"{prefix}.deterministic_algorithms_mode": torch.are_deterministic_algorithms_enabled(),
            f"{prefix}.deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        }
        logger.info(f"PyTorch config:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n")


@dataclass
class PyTorchCudaBackendState:
    allow_tf32: bool
    allow_fp16_reduced_precision_reduction: bool
    flash_sdp_enabled: bool
    math_sdp_enabled: bool
    preferred_linalg_backend: Any

    def restore(self) -> None:
        r"""Restores the PyTorch CUDA backend configuration by using the
        values in the state."""
        cuda.matmul.allow_tf32 = self.allow_tf32
        cuda.matmul.allow_fp16_reduced_precision_reduction = (
            self.allow_fp16_reduced_precision_reduction
        )
        cuda.enable_math_sdp(self.math_sdp_enabled)
        cuda.enable_flash_sdp(self.flash_sdp_enabled)
        cuda.preferred_linalg_library(self.preferred_linalg_backend)

    @classmethod
    def create(cls) -> PyTorchCudaBackendState:
        r"""Creates a state to capture the current PyTorch CUDA backend.

        Returns
        -------
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
    r"""Implements a context manager to configure the PyTorch CUDA
    backend.

    Args:
    ----
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import PyTorchCudaBackend
        >>> with PyTorchCudaBackend(allow_tf32=True):
        ...     pass
        ...
    """

    def __init__(
        self,
        allow_tf32: bool | None = None,
        allow_fp16_reduced_precision_reduction: bool | None = None,
        flash_sdp_enabled: bool | None = None,
        math_sdp_enabled: bool | None = None,
        preferred_linalg_backend: str | None = None,
        log_info: bool = False,
    ) -> None:
        self._allow_tf32 = allow_tf32
        self._allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        self._flash_sdp_enabled = flash_sdp_enabled
        self._math_sdp_enabled = math_sdp_enabled
        self._preferred_linalg_backend = preferred_linalg_backend

        self._log_info = bool(log_info)
        self._state: list[PyTorchCudaBackendState] = []

    def __enter__(self) -> PyTorchCudaBackend:
        logger.info("Configuring CUDA backend...")
        self._state.append(PyTorchCudaBackendState.create())
        self._configure()
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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
        logger.info(f"CUDA backend:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n")


@dataclass
class PyTorchCudnnBackendState:
    allow_tf32: bool
    benchmark: bool
    benchmark_limit: int | None
    deterministic: bool
    enabled: bool

    def restore(self) -> None:
        r"""Restores the PyTorch CUDNN backend configuration by using
        the values in the state."""
        cudnn.allow_tf32 = self.allow_tf32
        cudnn.benchmark = self.benchmark
        cudnn.benchmark_limit = self.benchmark_limit
        cudnn.deterministic = self.deterministic
        cudnn.enabled = self.enabled

    @classmethod
    def create(cls) -> PyTorchCudnnBackendState:
        r"""Creates a state to capture the current PyTorch CUDNN backend.

        Returns
        -------
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
    r"""Implements a context manager to configure the PyTorch CUDNN
    backend.

    Args:
    ----
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

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import PyTorchCudnnBackend
        >>> with PyTorchCudnnBackend(allow_tf32=True):
        ...     pass
        ...
    """

    def __init__(
        self,
        allow_tf32: bool = None,
        benchmark: bool | None = None,
        benchmark_limit: int | None = None,
        deterministic: bool | None = None,
        enabled: bool | None = None,
        log_info: bool = False,
    ) -> None:
        self._allow_tf32 = allow_tf32
        self._benchmark = benchmark
        self._benchmark_limit = benchmark_limit
        self._deterministic = deterministic
        self._enabled = enabled

        self._log_info = bool(log_info)
        self._state: list[PyTorchCudnnBackendState] = []

    def __enter__(self) -> PyTorchCudnnBackend:
        logger.info("Configuring CUDNN backend...")
        self._state.append(PyTorchCudnnBackendState.create())
        self._configure()
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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
        logger.info(f"CUDNN backend:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n")


@dataclass
class PyTorchMpsBackendState:
    is_available: bool
    is_built: bool

    def restore(self) -> None:
        r"""Restores the PyTorch MPS backend configuration by using the
        values in the state."""

    @classmethod
    def create(cls) -> PyTorchMpsBackendState:
        r"""Creates a state to capture the current PyTorch MPS backend.

        Returns
        -------
            ``PyTorchMpsBackendState``: The current state.
        """
        return cls(is_available=mps.is_available(), is_built=mps.is_built())


class PyTorchMpsBackend(BaseResource):
    r"""Implements a context manager to configure the PyTorch MPS
    backend.

    Args:
    ----
        log_info (bool, optional): If ``True``, the state is shown
            after the context manager is created. Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.rsrc import PyTorchMpsBackend
        >>> with PyTorchMpsBackend():
        ...     pass
        ...
    """

    def __init__(self, log_info: bool = False) -> None:
        self._log_info = bool(log_info)
        self._state: list[PyTorchMpsBackendState] = []

    def __enter__(self) -> PyTorchMpsBackend:
        logger.info("Configuring MPS backend...")
        self._state.append(PyTorchMpsBackendState.create())
        if self._log_info:
            self._show()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("Restoring MPS backend configuration...")
        self._state.pop().restore()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(log_info={self._log_info})"

    def _show(self) -> None:
        prefix = "torch.backends.mps"
        info = {
            f"{prefix}.is_available": mps.is_available(),
            f"{prefix}.is_built": mps.is_built(),
        }
        logger.info(f"MPS backend:\n{str_pretty_dict(info, sorted_keys=True, indent=2)}\n")
