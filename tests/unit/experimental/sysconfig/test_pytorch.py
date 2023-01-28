import logging
from unittest.mock import patch

from pytest import LogCaptureFixture, mark
from torch.backends import cuda

from gravitorch.experimental.sysconfig import PyTorchCudaBackend, pytorch_cuda_backend

########################################
#     Tests for PyTorchCudaBackend     #
########################################


def test_pytorch_cuda_backend_str():
    assert str(PyTorchCudaBackend()).startswith("PyTorchCudaBackend(")


@mark.parametrize("allow_tf32", (True, False))
def test_pytorch_cuda_backend_configure_allow_tf32(allow_tf32: bool):
    with pytorch_cuda_backend():
        PyTorchCudaBackend(allow_tf32=allow_tf32).configure()
        assert cuda.matmul.allow_tf32 == allow_tf32


@mark.parametrize("allow_fp16_reduced_precision_reduction", (True, False))
def test_pytorch_cuda_backend_configure_allow_fp16_reduced_precision_reduction(
    allow_fp16_reduced_precision_reduction: bool,
):
    with pytorch_cuda_backend():
        PyTorchCudaBackend(
            allow_fp16_reduced_precision_reduction=allow_fp16_reduced_precision_reduction
        ).configure()
        assert (
            cuda.matmul.allow_fp16_reduced_precision_reduction
            == allow_fp16_reduced_precision_reduction
        )


@mark.parametrize("flash_sdp_enabled", (True, False))
def test_pytorch_cuda_backend_configure_flash_sdp_enabled(flash_sdp_enabled: bool):
    with pytorch_cuda_backend():
        PyTorchCudaBackend(flash_sdp_enabled=flash_sdp_enabled).configure()
        assert cuda.flash_sdp_enabled() == flash_sdp_enabled


@mark.parametrize("math_sdp_enabled", (True, False))
def test_pytorch_cuda_backend_configure_math_sdp_enabled(math_sdp_enabled: bool):
    with pytorch_cuda_backend():
        PyTorchCudaBackend(math_sdp_enabled=math_sdp_enabled).configure()
        assert cuda.math_sdp_enabled() == math_sdp_enabled


@mark.parametrize("preferred_linalg_backend", ("cusolver", "magma", "default"))
def test_pytorch_cuda_backend_configure_preferred_linalg_backend(preferred_linalg_backend: str):
    sysconfig = PyTorchCudaBackend(preferred_linalg_backend=preferred_linalg_backend)
    with patch("torch.backends.cuda.preferred_linalg_library") as mock:
        sysconfig.configure()
        mock.assert_called_once_with(preferred_linalg_backend)


def test_pytorch_cuda_backend_show(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        PyTorchCudaBackend().show()
        assert len(caplog.messages[0]) > 0  # The message should not be empty
