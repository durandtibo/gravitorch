from typing import Optional
from unittest.mock import patch

from pytest import mark, raises

from gravitorch.distributed import Backend, UnknownBackendError, auto_backend
from gravitorch.distributed import backend as dist_backend
from gravitorch.distributed import (
    distributed_context,
    gloocontext,
    is_distributed,
    is_main_process,
    ncclcontext,
    resolve_backend,
)
from gravitorch.distributed.comm import BACKEND_TO_CONTEXT

#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process_true() -> None:
    # By definition, a non-distributed process is the main process.
    assert is_main_process()


@patch("gravitorch.distributed.comm.get_rank", lambda *args: 3)
def test_is_main_process_false() -> None:
    assert not is_main_process()


####################################
#     Tests for is_distributed     #
####################################


@patch("gravitorch.distributed.comm.get_world_size", lambda *args: 3)
def test_is_distributed_true() -> None:
    assert is_distributed()


def test_is_distributed_false() -> None:
    assert not is_distributed()


#########################################
#     Tests for distributed_context     #
#########################################


@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO,))
def test_distributed_context_backend() -> None:
    with patch("gravitorch.distributed.comm.initialize") as initialize_mock:
        with patch("gravitorch.distributed.comm.finalize") as finalize_mock:
            with distributed_context(Backend.GLOO):
                pass
    initialize_mock.assert_called_once_with(Backend.GLOO, init_method="env://")
    finalize_mock.assert_called_once_with()


def test_distributed_context_backend_incorrect() -> None:
    with raises(UnknownBackendError), distributed_context(backend="incorrect backend"):
        pass


def test_distributed_context_backend_raise_error() -> None:
    # Test if the `finalize` function is called to release the resources.
    with raises(RuntimeError), distributed_context(backend=Backend.GLOO):
        raise RuntimeError("Fake error")
    assert dist_backend() is None


##################################
#     Tests for auto_backend     #
##################################


@mark.parametrize("cuda_is_available", (True, False))
@patch("gravitorch.distributed.comm.available_backends", lambda *args: ())
def test_auto_backend_no_backend(cuda_is_available: bool) -> None:
    with patch("torch.cuda.is_available", lambda *args: cuda_is_available):
        assert auto_backend() is None


@patch("torch.cuda.is_available", lambda *args: False)
def test_auto_backend_no_gpu() -> None:
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: False)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_no_gpu_and_nccl() -> None:
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO,))
def test_auto_backend_gpu_and_no_nccl() -> None:
    assert auto_backend() == Backend.GLOO


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO, Backend.NCCL))
def test_auto_backend_gpu_and_nccl() -> None:
    assert auto_backend() == Backend.NCCL


#####################################
#     Tests for resolve_backend     #
#####################################


@mark.parametrize("backend", (Backend.GLOO, Backend.NCCL, None))
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO, Backend.NCCL))
def test_resolve_backend(backend: Optional[str]) -> None:
    assert resolve_backend(backend) == backend


@patch("gravitorch.distributed.comm.is_distributed_ready", lambda *args: False)
def test_resolve_backend_auto_should_not_initialize() -> None:
    assert resolve_backend("auto") is None


@mark.parametrize("backend", (Backend.GLOO, Backend.NCCL))
@patch("gravitorch.distributed.comm.is_distributed_ready", lambda *args: True)
def test_resolve_backend_auto_should_initialize(backend: str) -> None:
    with patch("gravitorch.distributed.comm.auto_backend", lambda *args: backend):
        assert resolve_backend("auto") == backend


def test_resolve_backend_incorrect_backend() -> None:
    with raises(UnknownBackendError):
        resolve_backend("incorrect")


#################################
#     Tests for gloocontext     #
#################################


@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.GLOO,))
def test_gloocontext() -> None:
    with patch("gravitorch.distributed.comm.distributed_context") as mock, gloocontext():
        mock.assert_called_once_with(Backend.GLOO)


@patch("gravitorch.distributed.comm.available_backends", lambda *args: ())
def test_gloocontext_no_gloo_backend() -> None:
    with raises(RuntimeError), gloocontext():
        pass


#################################
#     Tests for ncclcontext     #
#################################


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.NCCL,))
@patch("gravitorch.distributed.comm.get_local_rank", lambda *args: 1)
def test_ncclcontext() -> None:
    with patch("gravitorch.distributed.comm.distributed_context") as mock:
        with patch("gravitorch.distributed.comm.torch.cuda.device") as device:
            with ncclcontext():
                mock.assert_called_once_with(Backend.NCCL)
                device.assert_called_once_with(1)


@patch("torch.cuda.is_available", lambda *args: True)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: ())
def test_ncclcontext_no_nccl_backend() -> None:
    with raises(RuntimeError), ncclcontext():
        pass


@patch("torch.cuda.is_available", lambda *args: False)
@patch("gravitorch.distributed.comm.available_backends", lambda *args: (Backend.NCCL,))
def test_ncclcontext_cuda_is_not_available() -> None:
    with raises(RuntimeError), ncclcontext():
        pass


########################################
#     Tests for BACKEND_TO_CONTEXT     #
########################################


def test_backend_to_context() -> None:
    assert len(BACKEND_TO_CONTEXT) == 3
