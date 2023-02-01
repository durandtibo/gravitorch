r"""Defines some utility functions that are used to test some
functionalities."""
import torch
from ignite import distributed as idist
from pytest import fixture, mark

from gravitorch.distributed import comm as dist
from gravitorch.utils.integrations import (
    is_accelerate_available,
    is_pillow_available,
    is_tensorboard_available,
    is_torchvision_available,
)

cuda_available = mark.skipif(not torch.cuda.is_available(), reason="Requires a device with CUDA")
multi_gpus = mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
distributed_available = mark.skipif(
    not torch.distributed.is_available(), reason="Requires PyTorch distributed"
)
nccl_available = mark.skipif(
    dist.Backend.NCCL not in dist.available_backends(), reason="Requires NCCL"
)
gloo_available = mark.skipif(
    dist.Backend.GLOO not in dist.available_backends(), reason="Requires GLOO"
)

accelerate_available = mark.skipif(
    not is_accelerate_available(),
    reason=(
        "`accelerate` is not available. Please install `accelerate` if you want to run this test"
    ),
)
pillow_available = mark.skipif(
    not is_pillow_available(),
    reason="`pillow` is not available. Please install `pillow` if you want to run this test",
)
psutil_available = mark.skipif(
    not is_pillow_available(),
    reason="`psutil` is not available. Please install `psutil` if you want to run this test",
)
tensorboard_available = mark.skipif(
    not is_tensorboard_available(),
    reason=(
        "`tensorboard` is not available. Please install `tensorboard` if you want "
        "to run this test"
    ),
)
torchvision_available = mark.skipif(
    not is_torchvision_available(),
    reason=(
        "`torchvision` is not available. Please install `torchvision` if you want "
        "to run this test"
    ),
)

ignore_pytorch_lr_scheduler_warning = mark.filterwarnings("ignore::UserWarning")


@fixture
def parallel_gloo_2() -> idist.Parallel:
    return idist.Parallel(
        backend=dist.Backend.GLOO,
        nproc_per_node=2,
        nnodes=1,
        master_addr="127.0.0.1",
        master_port=29500,
    )


@fixture
def parallel_nccl_2() -> idist.Parallel:
    return idist.Parallel(
        backend=dist.Backend.NCCL,
        nproc_per_node=2,
        nnodes=1,
        master_addr="127.0.0.1",
        master_port=29500,
    )
