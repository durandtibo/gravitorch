__all__ = ["PyTorchCudaBackend", "pytorch_cuda_backend"]

import logging
from contextlib import contextmanager
from typing import Optional

from torch.backends import cuda

from gravitorch.experimental.sysconfig.base import BaseSysConfig
from gravitorch.utils.format import to_pretty_dict_str

logger = logging.getLogger(__name__)


class PyTorchCudaBackend(BaseSysConfig):
    r"""Configure the PyTorch CUDA backend.

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
    """

    def __init__(
        self,
        allow_tf32: Optional[bool] = None,
        allow_fp16_reduced_precision_reduction: Optional[bool] = None,
        flash_sdp_enabled: Optional[bool] = None,
        math_sdp_enabled: Optional[bool] = None,
        preferred_linalg_backend: Optional[str] = None,
    ):
        self._allow_tf32 = allow_tf32
        self._allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        self._flash_sdp_enabled = flash_sdp_enabled
        self._math_sdp_enabled = math_sdp_enabled
        self._preferred_linalg_backend = preferred_linalg_backend

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  allow_tf32={self._allow_tf32},\n"
            f"  allow_fp16_reduced_precision_reduction={self._allow_fp16_reduced_precision_reduction},\n"
            f"  flash_sdp_enabled={self._flash_sdp_enabled},\n"
            f"  math_sdp_enabled={self._math_sdp_enabled},\n"
            f"  preferred_linalg_backend={self._preferred_linalg_backend},\n"
            ")"
        )

    def configure(self) -> None:
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

    def show(self) -> None:
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
        }
        logger.info(
            f"PyTorch CUDA backend:\n{to_pretty_dict_str(info, sorted_keys=True, indent=2)}\n"
        )


@contextmanager
def pytorch_cuda_backend():
    r"""Defines a context manager to easily manage the PyTorch cuda backend
    configuration."""
    allow_tf32 = cuda.matmul.allow_tf32
    allow_fp16_reduced_precision_reduction = cuda.matmul.allow_fp16_reduced_precision_reduction
    math_sdp_enabled = cuda.math_sdp_enabled()
    flash_sdp_enabled = cuda.flash_sdp_enabled()
    preferred_linalg_library = cuda.preferred_linalg_library()
    try:
        yield
    finally:
        cuda.matmul.allow_tf32 = allow_tf32
        cuda.matmul.allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        cuda.enable_math_sdp(math_sdp_enabled)
        cuda.enable_flash_sdp(flash_sdp_enabled)
        cuda.preferred_linalg_library(preferred_linalg_library)
