from typing import Optional
from unittest.mock import patch

import torch
from pytest import mark, raises

from gravitorch.distributed import (
    Backend,
    UnknownBackendError,
    auto_backend,
    available_backends,
)
from gravitorch.distributed import backend as dist_backend
from gravitorch.distributed import (
    distributed_context,
    gloo,
    is_distributed,
    is_main_process,
    nccl,
    resolve_backend,
)

#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process_true():
    # By definition, a non-distributed process is the main process.
    assert is_main_process()


@patch("gravitorch.distributed.comm.get_rank", lambda *args: 3)
def test_is_main_process_false():
    assert not is_main_process()


####################################
#     Tests for is_distributed     #
####################################


@patch("gravitorch.distributed.comm.get_world_size", lambda *args: 3)
def test_is_distributed_true():
    assert is_distributed()


def test_is_distributed_false():
    assert not is_distributed()


#########################################
#     Tests for distributed_context     #
#########################################


@mark.parametrize("backend", available_backends())
def test_distributed_context_backend(backend):
    if backend == Backend.NCCL and not torch.cuda.is_available():
        return  # no cuda capable device
    with distributed_context(backend):
        assert dist_backend() == backend
    assert dist_backend() is None


def test_distributed_context_backend_incorrect():
    with raises(UnknownBackendError):
        with distributed_context(backend="incorrect backend"):
            pass


def test_distributed_context_backend_raise_error():
    # Test if the `finalize` function is called to release the resources.
    with raises(RuntimeError):
        with distributed_context(backend=Backend.GLOO):
            raise RuntimeError("Fake error")
    assert dist_backend() is None


##################################
#     Tests for auto_backend     #
##################################


@mark.parametrize("cuda_is_available", (True, False))
@patch("gravitorch.distributed.comm.available_backends", lambda *args: tuple())
def test_auto_backend_no_backend(cuda_is_available: bool):
    with patch("torch.cuda.is_available", lambda *args: cuda_is_available):
        assert auto_backend() is None


@patch("torch.cuda.is_available", lambda *args: False)
def test_auto_backend_no_gpu():
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: False)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_no_gpu_and_nccl():
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO,))
def test_auto_backend_gpu_and_no_nccl():
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_gpu_and_nccl():
    assert auto_backend() == Backend.NCCL


#####################################
#     Tests for resolve_backend     #
#####################################


@mark.parametrize("backend", (Backend.GLOO, Backend.NCCL, None))
def test_resolve_backend(backend: Optional[str]):
    assert resolve_backend(backend) == backend


@patch("gravitorch.distributed.comm.is_distributed_ready", lambda *args: False)
def test_resolve_backend_auto_should_not_initialize():
    assert resolve_backend("auto") is None


@mark.parametrize("backend", (Backend.GLOO, Backend.NCCL))
@patch("gravitorch.distributed.comm.is_distributed_ready", lambda *args: True)
def test_resolve_backend_auto_should_initialize(backend: str):
    with patch("gravitorch.distributed.comm.auto_backend", lambda *args: backend):
        assert resolve_backend("auto") == backend


##########################
#     Tests for gloo     #
##########################


@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO,))
def test_gloo():
    with patch("gravitorch.distributed.comm.distributed_context") as mock:
        with gloo():
            mock.assert_called_once_with(Backend.GLOO)


@patch("gravitorch.distributed.comm.available_backends", lambda *args: tuple())
def test_gloo_no_gloo_backend():
    with raises(RuntimeError):
        with gloo():
            pass


##########################
#     Tests for nccl     #
##########################


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.NCCL,))
@patch("gravitorch.distributed.comm.get_local_rank", lambda *args: 1)
def test_nccl():
    with patch("gravitorch.distributed.comm.distributed_context") as mock:
        with patch("gravitorch.distributed.comm.torch.cuda.device") as device:
            with nccl():
                mock.assert_called_once_with(Backend.NCCL)
                device.assert_called_once_with(1)


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: tuple())
def test_nccl_no_nccl_backend():
    with raises(RuntimeError):
        with nccl():
            pass


@patch("torch.cuda.is_available", lambda *args: False)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.NCCL,))
def test_nccl_cuda_is_not_available():
    with raises(RuntimeError):
        with nccl():
            pass
