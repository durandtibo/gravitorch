import logging
from unittest.mock import patch

import torch
from pytest import LogCaptureFixture, mark
from torch.backends import cuda, cudnn

from gravitorch.rsrc import PyTorchConfig, PyTorchCudaBackend, PyTorchCudnnBackend
from gravitorch.rsrc.pytorch import PyTorchCudaBackendState, PyTorchCudnnBackendState

###################################
#     Tests for PyTorchConfig     #
###################################


def test_pytorch_config_str():
    assert str(PyTorchConfig()).startswith("PyTorchConfig(")


@patch("torch.cuda.is_available", lambda *args: True)
@patch("torch.cuda.current_device", lambda *args: torch.device("cuda:0"))
@patch("torch.cuda.get_device_capability", lambda *args: (1, 2))
@patch("torch.cuda.get_device_name", lambda *args: "meow")
def test_pytorch_config_with_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchConfig():
            pass
        assert len(caplog.messages) == 6


@patch("torch.cuda.is_available", lambda *args: False)
def test_pytorch_config_without_cuda(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchConfig():
            pass
        assert len(caplog.messages) == 4


#############################################
#     Tests for PyTorchCudaBackendState     #
#############################################


def test_pytorch_cuda_backend_state_create():
    state = PyTorchCudaBackendState.create()
    assert isinstance(state.allow_tf32, bool)
    assert isinstance(state.allow_fp16_reduced_precision_reduction, bool)
    assert isinstance(state.flash_sdp_enabled, bool)
    assert isinstance(state.math_sdp_enabled, bool)
    assert isinstance(state.preferred_linalg_backend, torch._C._LinalgBackend)


def test_pytorch_cuda_backend_state_restore():
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


def test_pytorch_cuda_backend_str():
    assert str(PyTorchCudaBackend()).startswith("PyTorchCudaBackend(")


@mark.parametrize("allow_tf32", (True, False))
def test_pytorch_cuda_backend_allow_tf32(allow_tf32: bool):
    default = cuda.matmul.allow_tf32
    with PyTorchCudaBackend(allow_tf32=allow_tf32):
        assert cuda.matmul.allow_tf32 == allow_tf32
    assert cuda.matmul.allow_tf32 == default


@mark.parametrize("allow_fp16_reduced_precision_reduction", (True, False))
def test_pytorch_cuda_backend_allow_fp16_reduced_precision_reduction(
    allow_fp16_reduced_precision_reduction: bool,
):
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
def test_pytorch_cuda_backend_flash_sdp_enabled(flash_sdp_enabled: bool):
    default = cuda.flash_sdp_enabled()
    with PyTorchCudaBackend(flash_sdp_enabled=flash_sdp_enabled):
        assert cuda.flash_sdp_enabled() == flash_sdp_enabled
    assert cuda.flash_sdp_enabled() == default


@mark.parametrize("math_sdp_enabled", (True, False))
def test_pytorch_cuda_backend_math_sdp_enabled(math_sdp_enabled: bool):
    default = cuda.math_sdp_enabled()
    with PyTorchCudaBackend(math_sdp_enabled=math_sdp_enabled):
        assert cuda.math_sdp_enabled() == math_sdp_enabled
    assert cuda.math_sdp_enabled() == default


@mark.parametrize("preferred_linalg_backend", ("cusolver", "magma", "default"))
def test_pytorch_cuda_backend_preferred_linalg_backend(preferred_linalg_backend: str):
    with patch("torch.backends.cuda.preferred_linalg_library") as mock:
        with PyTorchCudaBackend(preferred_linalg_backend=preferred_linalg_backend):
            mock.assert_called()


@mark.parametrize("preferred_linalg_backend", ("cusolver", "magma", "default"))
def test_pytorch_cuda_backend_configure_preferred_linalg_backend(preferred_linalg_backend: str):
    with patch("torch.backends.cuda.preferred_linalg_library"):
        with PyTorchCudaBackend(preferred_linalg_backend=preferred_linalg_backend) as resource:
            with patch("torch.backends.cuda.preferred_linalg_library") as mock:
                resource._configure()
                mock.assert_called_once_with(preferred_linalg_backend)


def test_pytorch_cuda_backend_log_info_true(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchCudaBackend(log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_pytorch_cuda_backend_log_info_false(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchCudaBackend():
            pass
        assert len(caplog.messages) == 2


def test_pytorch_cuda_backend_reentrant():
    default = cuda.matmul.allow_tf32
    resource = PyTorchCudaBackend(allow_tf32=True)
    with resource:
        with resource:
            assert cuda.matmul.allow_tf32
    assert cuda.matmul.allow_tf32 == default


##############################################
#     Tests for PyTorchCudnnBackendState     #
##############################################


def test_pytorch_cudnn_backend_state_create():
    state = PyTorchCudnnBackendState.create()
    assert isinstance(state.allow_tf32, bool)
    assert isinstance(state.benchmark, bool)
    assert isinstance(state.benchmark_limit, (int, type(None)))
    assert isinstance(state.deterministic, bool)
    assert isinstance(state.enabled, bool)


def test_pytorch_cudnn_backend_state_restore():
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


def test_pytorch_cudnn_backend_str():
    assert str(PyTorchCudnnBackend()).startswith("PyTorchCudnnBackend(")


@mark.parametrize("allow_tf32", (True, False))
def test_pytorch_cudnn_backend_allow_tf32(allow_tf32: bool):
    default = cudnn.allow_tf32
    with PyTorchCudnnBackend(allow_tf32=allow_tf32):
        assert cudnn.allow_tf32 == allow_tf32
    assert cudnn.allow_tf32 == default


@mark.parametrize("benchmark", (True, False))
def test_pytorch_cudnn_backend_benchmark(benchmark: bool):
    default = cudnn.benchmark
    with PyTorchCudnnBackend(benchmark=benchmark):
        assert cudnn.benchmark == benchmark
    assert cudnn.benchmark == default


@mark.parametrize("benchmark_limit", (0, 1))
def test_pytorch_cudnn_backend_benchmark_limit(benchmark_limit: int):
    default = cudnn.benchmark_limit
    with PyTorchCudnnBackend(benchmark_limit=benchmark_limit):
        assert cudnn.benchmark_limit == benchmark_limit
    assert cudnn.benchmark_limit == default


@mark.parametrize("deterministic", (True, False))
def test_pytorch_cudnn_backend_deterministic(deterministic: bool):
    default = cudnn.deterministic
    with PyTorchCudnnBackend(deterministic=deterministic):
        assert cudnn.deterministic == deterministic
    assert cudnn.deterministic == default


@mark.parametrize("enabled", (True, False))
def test_pytorch_cudnn_backend_enabled(enabled: bool):
    default = cudnn.enabled
    with PyTorchCudnnBackend(enabled=enabled):
        assert cudnn.enabled == enabled
    assert cudnn.enabled == default


def test_pytorch_cudnn_backend_log_info_true(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchCudnnBackend(log_info=True):
            pass
        assert len(caplog.messages) == 3


def test_pytorch_cudnn_backend_log_info_false(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        with PyTorchCudnnBackend():
            pass
        assert len(caplog.messages) == 2


def test_pytorch_cudnn_backend_reentrant():
    default = cudnn.allow_tf32
    resource = PyTorchCudnnBackend(allow_tf32=True)
    with resource:
        with resource:
            assert cudnn.allow_tf32
    assert cudnn.allow_tf32 == default
