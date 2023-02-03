from gravitorch.distributed import Backend
from gravitorch.distributed import backend as dist_backend
from gravitorch.rsrc.distributed import DistributedContext
from tests.testing import (
    cuda_available,
    distributed_available,
    gloo_available,
    nccl_available,
)

########################################
#     Tests for DistributedContext     #
########################################


def test_distributed_context_backend_none():
    with DistributedContext(backend=None):
        assert dist_backend() is None


@distributed_available
@gloo_available
def test_distributed_context_backend_gloo():
    with DistributedContext(backend=Backend.GLOO):
        assert dist_backend() == Backend.GLOO


@distributed_available
@cuda_available
@nccl_available
def test_distributed_context_backend_nccl():
    with DistributedContext(backend=Backend.NCCL):
        assert dist_backend() == Backend.NCCL
