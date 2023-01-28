__all__ = ["PyTorchCudaBackend", "PyTorchCudaBackendState"]

import logging
from dataclasses import dataclass
from typing import Any, Optional

from torch.backends import cuda

from gravitorch.experimental.rsrc.base import BaseResourceManager
from gravitorch.utils.format import to_pretty_dict_str

logger = logging.getLogger(__name__)


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
            allow_fp16_reduced_precision_reduction=cuda.matmul.allow_fp16_reduced_precision_reduction,
            math_sdp_enabled=cuda.math_sdp_enabled(),
            flash_sdp_enabled=cuda.flash_sdp_enabled(),
            preferred_linalg_backend=cuda.preferred_linalg_library(),
        )


class PyTorchCudaBackend(BaseResourceManager):
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
        show_state (bool, optional): If ``True``, the state is shown
            after the context manager is created. Default: ``False``
    """

    def __init__(
        self,
        allow_tf32: Optional[bool] = None,
        allow_fp16_reduced_precision_reduction: Optional[bool] = None,
        flash_sdp_enabled: Optional[bool] = None,
        math_sdp_enabled: Optional[bool] = None,
        preferred_linalg_backend: Optional[str] = None,
        show_state: bool = False,
    ):
        self._allow_tf32 = allow_tf32
        self._allow_fp16_reduced_precision_reduction = allow_fp16_reduced_precision_reduction
        self._flash_sdp_enabled = flash_sdp_enabled
        self._math_sdp_enabled = math_sdp_enabled
        self._preferred_linalg_backend = preferred_linalg_backend

        self._show_state = bool(show_state)
        self._state: list[PyTorchCudaBackendState] = []

    def __enter__(self):
        self._state.append(PyTorchCudaBackendState.create())
        self.configure()
        if self._show_state:
            self.show()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state.pop().restore()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  allow_tf32={self._allow_tf32},\n"
            f"  allow_fp16_reduced_precision_reduction={self._allow_fp16_reduced_precision_reduction},\n"
            f"  flash_sdp_enabled={self._flash_sdp_enabled},\n"
            f"  math_sdp_enabled={self._math_sdp_enabled},\n"
            f"  preferred_linalg_backend={self._preferred_linalg_backend},\n"
            f"  show_state={self._show_state},\n"
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
        logger.info(f"CUDA backend:\n{to_pretty_dict_str(info, sorted_keys=True, indent=2)}\n")
