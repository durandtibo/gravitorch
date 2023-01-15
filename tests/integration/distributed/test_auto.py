from unittest.mock import patch

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gravitorch import distributed as dist
from gravitorch.distributed.auto import auto_ddp_model
from tests.testing import cuda_available, gloo_available, nccl_available


@gloo_available
def test_auto_distributed_context_dist_backend_gloo():
    with dist.auto_distributed_context(dist_backend=dist.Backend.GLOO):
        assert dist.backend() == dist.Backend.GLOO


@nccl_available
@cuda_available
def test_auto_distributed_context_dist_backend_nccl():
    with dist.auto_distributed_context(dist_backend=dist.Backend.NCCL):
        assert dist.backend() == dist.Backend.NCCL
        device = dist.device()
        assert device.type == "cuda"
        assert device.index == dist.get_local_rank()


@gloo_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_gloo_backend():
    with dist.auto_distributed_context(dist_backend=dist.Backend.GLOO):
        model = nn.Linear(4, 6)
        model = auto_ddp_model(model)
        assert isinstance(model, DistributedDataParallel)


@nccl_available
@cuda_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_nccl_backend():
    with dist.auto_distributed_context(dist_backend=dist.Backend.NCCL):
        model = nn.Linear(4, 6)
        model = auto_ddp_model(model)
        assert isinstance(model, DistributedDataParallel)
