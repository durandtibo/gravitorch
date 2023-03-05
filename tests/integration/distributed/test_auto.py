from unittest.mock import patch

from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gravitorch.distributed import gloocontext, ncclcontext
from gravitorch.distributed.auto import auto_ddp_model
from gravitorch.testing import cuda_available, gloo_available, nccl_available

####################################
#     Tests for auto_ddp_model     #
####################################


@gloo_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_gloo_backend() -> None:
    with gloocontext():
        model = nn.Linear(4, 6)
        model = auto_ddp_model(model)
        assert isinstance(model, DistributedDataParallel)


@nccl_available
@cuda_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_nccl_backend() -> None:
    with ncclcontext():
        model = nn.Linear(4, 6)
        model = auto_ddp_model(model)
        assert isinstance(model, DistributedDataParallel)
