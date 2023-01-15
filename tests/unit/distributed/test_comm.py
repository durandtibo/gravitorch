from unittest.mock import patch

import torch
from pytest import mark, raises

from gravitorch.distributed import comm


def test_is_main_process_true():
    # By definition, a non-distributed process is the main process.
    assert comm.is_main_process()


@patch("gravitorch.distributed.comm.get_rank", lambda *args: 3)
def test_is_main_process_false():
    assert not comm.is_main_process()


@patch("gravitorch.distributed.comm.get_world_size", lambda *args: 3)
def test_is_distributed_true():
    assert comm.is_distributed()


def test_is_distributed_false():
    assert not comm.is_distributed()


@mark.parametrize("backend", comm.available_backends())
def test_setup_distributed_context_backend(backend):
    if backend == comm.Backend.NCCL and not torch.cuda.is_available():
        return  # no cuda capable device
    with comm.setup_distributed_context(backend):
        assert comm.backend() == backend
    assert comm.backend() is None


def test_setup_distributed_context_backend_incorrect():
    with raises(comm.UnknownBackendError):
        with comm.setup_distributed_context(backend="incorrect backend"):
            pass


def test_setup_distributed_context_backend_raise_error():
    # Test if the `finalize` function is called to release the resources.
    with raises(RuntimeError):
        with comm.setup_distributed_context(backend=comm.Backend.GLOO):
            raise RuntimeError("Fake error")
    assert comm.backend() is None
