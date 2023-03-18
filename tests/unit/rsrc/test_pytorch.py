import logging
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch.backends import cuda, cudnn, mps

from gravitorch.rsrc import PyTorchConfig, PyTorchCudaBackend, PyTorchCudnnBackend
from gravitorch.rsrc.pytorch import (
    PyTorchConfigState,
    PyTorchCudaBackendState,
    PyTorchCudnnBackendState,
    PyTorchMpsBackend,
    PyTorchMpsBackendState,
)

########################################
#     Tests for PyTorchConfigState     #
########################################


def test_pytorch_config_state_create() -> None:
    state = PyTorchConfigState.create()
    assert state.float32_matmul_precision == torch.get_float32_matmul_precision()
    assert state.deterministic_algorithms_mode == torch.are_deterministic_algorithms_enabled()
    assert (
        state.deterministic_algorithms_warn_only
        == torch.is_deterministic_algorithms_warn_only_enabled()
    )


def test_pytorch_config_state_restore() -> None:
    with PyTorchConfig():
        PyTorchConfigState(
            float32_matmul_precision="high",
            deterministic_algorithms_mode=True,
            deterministic_algorithms_warn_only=True,
        ).restore()
        assert torch.get_float32_matmul_precision()
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()


###################################
#     Tests for PyTorchConfig     #
###################################


def test_pytorch_config_str() -> None:
    assert str(PyTorchConfig()).startswith("PyTorchConfig(")


@patch("torch.cuda.is_available", lambda *args: True)
@patch("torch.cuda.current_device", lambda *args: torch.device("cuda:0"))
@patch("torch.cuda.get_device_capability", lambda *args: (1, 2))
@patch("torch.cuda.get_device_name", lambda *args: "meow")
def test_pytorch_config_with_cuda(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchConfig():
            pass
        assert len(caplog.messages) == 8


@patch("torch.cuda.is_available", lambda *args: False)
def test_pytorch_config_without_cuda(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchConfig():
            pass
        assert len(caplog.messages) == 6


@mark.parametrize("float32_matmul_precision", ("highest", "high"))
def test_pytorch_config_float32_matmul_precision(float32_matmul_precision: str) -> None:
    default = torch.get_float32_matmul_precision()
    with PyTorchConfig(float32_matmul_precision=float32_matmul_precision):
        assert torch.get_float32_matmul_precision() == float32_matmul_precision
    assert torch.get_float32_matmul_precision() == default


@mark.parametrize("deterministic_algorithms_mode", (True, False))
@mark.parametrize("deterministic_algorithms_warn_only", (True, False))
def test_pytorch_config_deterministic_algorithms(
    deterministic_algorithms_mode: bool, deterministic_algorithms_warn_only: bool
) -> None:
    default_mode = torch.are_deterministic_algorithms_enabled()
    default_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with PyTorchConfig(
        deterministic_algorithms_mode=deterministic_algorithms_mode,
        deterministic_algorithms_warn_only=deterministic_algorithms_warn_only,
    ):
        assert torch.are_deterministic_algorithms_enabled() == deterministic_algorithms_mode
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled()
            == deterministic_algorithms_warn_only
        )
    assert torch.are_deterministic_algorithms_enabled() == default_mode
    assert torch.is_deterministic_algorithms_warn_only_enabled() == default_warn_only


@patch("torch.cuda.is_available", lambda *args: False)
def test_pytorch_config_log_info_true(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchConfig(log_info=True):
            pass
        assert len(caplog.messages) == 7


@patch("torch.cuda.is_available", lambda *args: False)
def test_pytorch_config_log_info_false(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchConfig():
            pass
        assert len(caplog.messages) == 6


def test_pytorch_config_reentrant() -> None:
    default = torch.get_float32_matmul_precision()
    resource = PyTorchConfig(float32_matmul_precision="high")
    with resource, resource:
        assert torch.get_float32_matmul_precision() == "high"
    assert torch.get_float32_matmul_precision() == default


#############################################
#     Tests for PyTorchCudaBackendState     #
#############################################


def test_pytorch_cuda_backend_state_create() -> None:
    state = PyTorchCudaBackendState.create()
    assert isinstance(state.allow_tf32, bool)
    assert isinstance(state.allow_fp16_reduced_precision_reduction, bool)
    assert isinstance(state.flash_sdp_enabled, bool)
    assert isinstance(state.math_sdp_enabled, bool)
    assert isinstance(state.preferred_linalg_backend, torch._C._LinalgBackend)


def test_pytorch_cuda_backend_state_restore() -> None:
    with PyTorchCudaBackend():
        PyTorchCudaBackendState(
            allow_tf32=True,
            allow_fp16_reduced_precision_reduction=True,
            flash_sdp_enabled=True,
            math_sdp_enabled=True,
            preferred_linalg_backend="default",
        ).restore()
        assert cuda.matmul.allow_tf32
        assert cuda.matmul.allow_fp16_reduced_precision_reduction
        assert cuda.flash_sdp_enabled()
        assert cuda.math_sdp_enabled()
        assert isinstance(cuda.preferred_linalg_library(), torch._C._LinalgBackend)


########################################
#     Tests for PyTorchCudaBackend     #
########################################


def test_pytorch_cuda_backend_str() -> None:
    assert str(PyTorchCudaBackend()).startswith("PyTorchCudaBackend(")


@mark.parametrize("allow_tf32", (True, False))
def test_pytorch_cuda_backend_allow_tf32(allow_tf32: bool) -> None:
    default = cuda.matmul.allow_tf32
    with PyTorchCudaBackend(allow_tf32=allow_tf32):
        assert cuda.matmul.allow_tf32 == allow_tf32
    assert cuda.matmul.allow_tf32 == default


@mark.parametrize("allow_fp16_reduced_precision_reduction", (True, False))
def test_pytorch_cuda_backend_allow_fp16_reduced_precision_reduction(
    allow_fp16_reduced_precision_reduction: bool,
) -> None:
    default = cuda.matmul.allow_fp16_reduced_precision_reduction
    with PyTorchCudaBackend(
        allow_fp16_reduced_precision_reduction=allow_fp16_reduced_precision_reduction
    ):
        assert (
            cuda.matmul.allow_fp16_reduced_precision_reduction
            == allow_fp16_reduced_precision_reduction
        )
    assert cuda.matmul.allow_fp16_reduced_precision_reduction == default


@mark.parametrize("flash_sdp_enabled", (True, False))
def test_pytorch_cuda_backend_flash_sdp_enabled(flash_sdp_enabled: bool) -> None:
    default = cuda.flash_sdp_enabled()
    with PyTorchCudaBackend(flash_sdp_enabled=flash_sdp_enabled):
        assert cuda.flash_sdp_enabled() == flash_sdp_enabled
    assert cuda.flash_sdp_enabled() == default


@mark.parametrize("math_sdp_enabled", (True, False))
def test_pytorch_cuda_backend_math_sdp_enabled(math_sdp_enabled: bool) -> None:
    default = cuda.math_sdp_enabled()
    with PyTorchCudaBackend(math_sdp_enabled=math_sdp_enabled):
        assert cuda.math_sdp_enabled() == math_sdp_enabled
    assert cuda.math_sdp_enabled() == default


@mark.parametrize("preferred_linalg_backend", ("cusolver", "magma", "default"))
def test_pytorch_cuda_backend_preferred_linalg_backend(preferred_linalg_backend: str) -> None:
    with patch("torch.backends.cuda.preferred_linalg_library") as mock:
        with PyTorchCudaBackend(preferred_linalg_backend=preferred_linalg_backend):
            mock.assert_called()


@mark.parametrize("preferred_linalg_backend", ("cusolver", "magma", "default"))
def test_pytorch_cuda_backend_configure_preferred_linalg_backend(
    preferred_linalg_backend: str,
) -> None:
    with patch("torch.backends.cuda.preferred_linalg_library"):
        with PyTorchCudaBackend(preferred_linalg_backend=preferred_linalg_backend) as resource:
            with patch("torch.backends.cuda.preferred_linalg_library") as mock:
                resource._configure()
                mock.assert_called_once_with(preferred_linalg_backend)


def test_pytorch_cuda_backend_log_info_true(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchCudaBackend(log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_pytorch_cuda_backend_log_info_false(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchCudaBackend():
            pass
        assert len(caplog.messages) == 2


def test_pytorch_cuda_backend_reentrant() -> None:
    default = cuda.matmul.allow_tf32
    resource = PyTorchCudaBackend(allow_tf32=True)
    with resource, resource:
        assert cuda.matmul.allow_tf32
    assert cuda.matmul.allow_tf32 == default


##############################################
#     Tests for PyTorchCudnnBackendState     #
##############################################


def test_pytorch_cudnn_backend_state_create() -> None:
    state = PyTorchCudnnBackendState.create()
    assert isinstance(state.allow_tf32, bool)
    assert isinstance(state.benchmark, bool)
    assert isinstance(state.benchmark_limit, (int, type(None)))
    assert isinstance(state.deterministic, bool)
    assert isinstance(state.enabled, bool)


def test_pytorch_cudnn_backend_state_restore() -> None:
    with PyTorchCudaBackend():
        PyTorchCudnnBackendState(
            allow_tf32=True,
            benchmark=True,
            benchmark_limit=0,
            deterministic=True,
            enabled=True,
        ).restore()
        assert cudnn.allow_tf32
        assert cudnn.benchmark
        assert cudnn.benchmark_limit == 0
        assert cudnn.deterministic
        assert cudnn.enabled


#########################################
#     Tests for PyTorchCudnnBackend     #
#########################################


def test_pytorch_cudnn_backend_str() -> None:
    assert str(PyTorchCudnnBackend()).startswith("PyTorchCudnnBackend(")


@mark.parametrize("allow_tf32", (True, False))
def test_pytorch_cudnn_backend_allow_tf32(allow_tf32: bool) -> None:
    default = cudnn.allow_tf32
    with PyTorchCudnnBackend(allow_tf32=allow_tf32):
        assert cudnn.allow_tf32 == allow_tf32
    assert cudnn.allow_tf32 == default


@mark.parametrize("benchmark", (True, False))
def test_pytorch_cudnn_backend_benchmark(benchmark: bool) -> None:
    default = cudnn.benchmark
    with PyTorchCudnnBackend(benchmark=benchmark):
        assert cudnn.benchmark == benchmark
    assert cudnn.benchmark == default


@mark.parametrize("benchmark_limit", (0, 1))
def test_pytorch_cudnn_backend_benchmark_limit(benchmark_limit: int) -> None:
    default = cudnn.benchmark_limit
    with PyTorchCudnnBackend(benchmark_limit=benchmark_limit):
        assert cudnn.benchmark_limit == benchmark_limit
    assert cudnn.benchmark_limit == default


@mark.parametrize("deterministic", (True, False))
def test_pytorch_cudnn_backend_deterministic(deterministic: bool) -> None:
    default = cudnn.deterministic
    with PyTorchCudnnBackend(deterministic=deterministic):
        assert cudnn.deterministic == deterministic
    assert cudnn.deterministic == default


@mark.parametrize("enabled", (True, False))
def test_pytorch_cudnn_backend_enabled(enabled: bool) -> None:
    default = cudnn.enabled
    with PyTorchCudnnBackend(enabled=enabled):
        assert cudnn.enabled == enabled
    assert cudnn.enabled == default


def test_pytorch_cudnn_backend_log_info_true(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchCudnnBackend(log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_pytorch_cudnn_backend_log_info_false(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchCudnnBackend():
            pass
        assert len(caplog.messages) == 2


def test_pytorch_cudnn_backend_reentrant() -> None:
    default = cudnn.allow_tf32
    resource = PyTorchCudnnBackend(allow_tf32=True)
    with resource, resource:
        assert cudnn.allow_tf32
    assert cudnn.allow_tf32 == default


############################################
#     Tests for PyTorchMpsBackendState     #
############################################


def test_pytorch_mps_backend_state_create() -> None:
    state = PyTorchMpsBackendState.create()
    assert state.is_available == mps.is_available()
    assert state.is_built == mps.is_built()


def test_pytorch_mps_backend_state_restore() -> None:
    with PyTorchMpsBackend():
        PyTorchMpsBackendState(is_available=False, is_built=False).restore()


#######################################
#     Tests for PyTorchMpsBackend     #
#######################################


def test_pytorch_mps_backend_str() -> None:
    assert str(PyTorchMpsBackend()).startswith("PyTorchMpsBackend(")


def test_pytorch_mps_backend_log_info_true(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchMpsBackend(log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_pytorch_mps_backend_log_info_false(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with PyTorchMpsBackend():
            pass
        assert len(caplog.messages) == 2


def test_pytorch_mps_backend_reentrant() -> None:
    resource = PyTorchMpsBackend()
    with resource, resource:
        assert isinstance(mps.is_available(), bool)
    assert isinstance(mps.is_available(), bool)
