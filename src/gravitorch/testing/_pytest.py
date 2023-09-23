from __future__ import annotations

import torch
from pytest import mark
from torch import distributed as tdist

from gravitorch.distributed.comm import Backend, available_backends
from gravitorch.utils.imports import (
    is_accelerate_available,
    is_matplotlib_available,
    is_pillow_available,
    is_tensorboard_available,
    is_torchdata_available,
    is_torchvision_available,
    is_tqdm_available,
)

cuda_available = mark.skipif(not torch.cuda.is_available(), reason="Requires a device with CUDA")
two_gpus_available = mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
distributed_available = mark.skipif(not tdist.is_available(), reason="Requires PyTorch distributed")
nccl_available = mark.skipif(Backend.NCCL not in available_backends(), reason="Requires NCCL")
gloo_available = mark.skipif(Backend.GLOO not in available_backends(), reason="Requires GLOO")

accelerate_available = mark.skipif(
    not is_accelerate_available(),
    reason=(
        "`accelerate` is not available. Please install `accelerate` if you want to run this test"
    ),
)
matplotlib_available = mark.skipif(
    not is_matplotlib_available(),
    reason=(
        "`matplotlib` is not available. Please install `matplotlib` if you want to run this test"
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
torchdata_available = mark.skipif(
    not is_torchdata_available(),
    reason=(
        "`torchdata` is not available. Please install `torchdata` if you want to run this test"
    ),
)
torchvision_available = mark.skipif(
    not is_torchvision_available(),
    reason=(
        "`torchvision` is not available. Please install `torchvision` if you want "
        "to run this test"
    ),
)
tqdm_available = mark.skipif(
    not is_tqdm_available(),
    reason="`tqdm` is not available. Please install `tqdm` if you want to run this test",
)
