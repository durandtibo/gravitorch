from unittest.mock import Mock, patch

import torch
from objectory import OBJECT_TARGET
from pytest import fixture
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gravitorch.creators.model import BaseModelCreator, VanillaModelCreator
from gravitorch.creators.model.ddp import DataDistributedParallelModelCreator, to_ddp
from gravitorch.distributed import gloocontext, ncclcontext
from gravitorch.nn import get_module_device
from tests.testing import cuda_available, gloo_available, nccl_available

#########################################################
#     Tests for DataDistributedParallelModelCreator     #
#########################################################


@fixture
def model_creator() -> BaseModelCreator:
    return VanillaModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
    )


@gloo_available
def test_data_distributed_parallel_model_creator_create_gloo(model_creator):
    with gloocontext():
        creator = DataDistributedParallelModelCreator(model_creator=model_creator)
        model = creator.create(engine=Mock())
        assert isinstance(model, DistributedDataParallel)
        assert isinstance(model.module, nn.Linear)


@cuda_available
@nccl_available
@patch("gravitorch.creators.model.ddp.dist.get_world_size", lambda *args: 2)
def test_data_distributed_parallel_model_creator_create_nccl(model_creator):
    with ncclcontext():
        creator = DataDistributedParallelModelCreator(model_creator=model_creator)
        model = creator.create(engine=Mock())
        assert isinstance(model, DistributedDataParallel)
        assert isinstance(model.module, nn.Linear)


############################
#     Tests for to_ddp     #
############################


@gloo_available
def test_to_ddp_already_ddp():
    with gloocontext():
        module = to_ddp(DistributedDataParallel(nn.Linear(4, 5)))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)


@gloo_available
def test_to_ddp_linear_gloo():
    with gloocontext():
        module = to_ddp(nn.Linear(4, 5))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)


@cuda_available
@nccl_available
@patch("gravitorch.creators.model.ddp.dist.get_world_size", lambda *args: 2)
def test_to_ddp_linear_nccl():
    with ncclcontext():
        module = to_ddp(nn.Linear(4, 5).to(device=torch.device("cuda:0")))
        assert isinstance(module, DistributedDataParallel)
        assert isinstance(module.module, nn.Linear)
        assert get_module_device(module) == torch.device("cuda:0")
