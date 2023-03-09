from unittest.mock import patch

import torch
from pytest import mark
from torch import Tensor, nn

from gravitorch.distributed import comm as dist
from gravitorch.distributed.auto import (
    _manage_model_device,
    _wrap_distributed_data_parallel,
    auto_ddp_model,
)
from gravitorch.nn import get_module_device, get_module_devices, is_module_on_device
from gravitorch.testing import cuda_available, distributed_available, nccl_available
from gravitorch.utils import get_available_devices

##########################
#     auto_ddp_model     #
##########################


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 5)
        self.bn = nn.BatchNorm1d(5)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.bn(self.fc(tensor.to(device=get_module_devices(self.fc)[0])))


@patch("gravitorch.distributed.auto.dist.device", lambda *args: torch.device("cpu"))
def test_auto_ddp_model_cpu() -> None:
    device = torch.device("cpu")
    model = nn.Linear(4, 5).to(device=device)
    model = auto_ddp_model(model)
    # Verify all the parameters are on the CPU device.
    assert all([p.device == device for p in model.parameters()])


@patch("gravitorch.distributed.auto.dist.device", lambda *args: torch.device("cuda"))
@cuda_available
def test_auto_ddp_model_move_cuda() -> None:
    model = nn.Linear(4, 5).to(device=torch.device("cpu"))
    model = auto_ddp_model(model)
    # Verify all the parameters are on the CUDA device.
    assert all([p.device == torch.device("cuda:0") for p in model.parameters()])


@distributed_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_gloo() -> None:
    with dist.distributed_context(backend=dist.Backend.GLOO):
        assert isinstance(auto_ddp_model(FakeModel()), nn.parallel.DistributedDataParallel)


@distributed_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_gloo_sync_bn() -> None:
    with dist.distributed_context(backend=dist.Backend.GLOO):
        model = FakeModel()
        model = auto_ddp_model(model, sync_batch_norm=True)
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        # The conversion should be done only for NCCL backend, not for the GLOO backend.
        assert isinstance(model.module.bn, nn.BatchNorm1d)


@cuda_available
@nccl_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_nccl() -> None:
    with dist.distributed_context(backend=dist.Backend.NCCL):
        model = auto_ddp_model(FakeModel())
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert get_module_device(model) == torch.device("cuda:0")


@cuda_available
@nccl_available
@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_nccl_sync_bn() -> None:
    with dist.distributed_context(backend=dist.Backend.NCCL):
        model = FakeModel()
        model = auto_ddp_model(model, sync_batch_norm=True)
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert isinstance(model.module.bn, nn.SyncBatchNorm)


@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_no_learnable_parameters() -> None:
    with dist.distributed_context(backend=dist.Backend.GLOO):
        assert isinstance(auto_ddp_model(nn.Tanh()), nn.Tanh)


@patch("gravitorch.distributed.auto.dist.get_world_size", lambda *args: 2)
def test_auto_ddp_model_ddp_distributed_data_parallel() -> None:
    with dist.distributed_context(backend=dist.Backend.GLOO):
        model = nn.parallel.DistributedDataParallel(nn.Linear(4, 5))
        model = auto_ddp_model(model)
        # It should not wrap the model with ``DistributedDataParallel`` because it is already a
        # ``DistributedDataParallel`` module.
        assert isinstance(model.module, nn.Linear)


##########################################
#     Tests for _manage_model_device     #
##########################################


@mark.parametrize("device", get_available_devices())
def test_manage_model_device_same_device(device: str) -> None:
    device = torch.device(device)
    model = nn.Linear(4, 6).to(device=device)
    with patch("gravitorch.distributed.auto.dist.device", lambda *args, **kwargs: device):
        model = _manage_model_device(model)
    assert is_module_on_device(model, device)


@mark.parametrize("device", get_available_devices())
def test_manage_model_device_different_device(device: str) -> None:
    device = torch.device(device)
    model = nn.Linear(4, 6).cpu()
    with patch("gravitorch.distributed.auto.dist.device", lambda *args, **kwargs: device), patch(
        "gravitorch.distributed.auto.is_module_on_device", lambda *args, **kwargs: False
    ):
        model = _manage_model_device(model)
    assert is_module_on_device(model, device)


#####################################################
#     Tests for _wrap_distributed_data_parallel     #
#####################################################


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: "UNKNOWN_BACKEND")
def test_wrap_distributed_data_parallel_unknown_backend() -> None:
    assert isinstance(_wrap_distributed_data_parallel(nn.Linear(4, 6)), nn.Linear)


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: dist.Backend.GLOO)
def test_wrap_distributed_data_parallel_gloo_backend() -> None:
    with patch("gravitorch.distributed.auto.DistributedDataParallel") as mocked_ddp:
        model = nn.Linear(4, 6)
        _wrap_distributed_data_parallel(model)
        mocked_ddp.assert_called_once_with(model)


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: dist.Backend.GLOO)
def test_wrap_distributed_data_parallel_gloo_backend_extra_args() -> None:
    with patch("gravitorch.distributed.auto.DistributedDataParallel") as mocked_ddp:
        model = nn.Linear(4, 6)
        _wrap_distributed_data_parallel(model, find_unused_parameters=True)
        mocked_ddp.assert_called_once_with(model, find_unused_parameters=True)


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: dist.Backend.NCCL)
@patch("gravitorch.distributed.auto.dist.get_local_rank", lambda *args, **kwargs: 1)
def test_wrap_distributed_data_parallel_nccl_backend() -> None:
    with patch("gravitorch.distributed.auto.DistributedDataParallel") as mocked_ddp:
        model = nn.Linear(4, 6)
        _wrap_distributed_data_parallel(model)
        mocked_ddp.assert_called_once_with(model, device_ids=[1])


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: dist.Backend.NCCL)
@patch("gravitorch.distributed.auto.dist.get_local_rank", lambda *args, **kwargs: 1)
def test_wrap_distributed_data_parallel_nccl_backend_extra_args() -> None:
    with patch("gravitorch.distributed.auto.DistributedDataParallel") as mocked_ddp:
        model = nn.Linear(4, 6)
        _wrap_distributed_data_parallel(model, find_unused_parameters=True)
        mocked_ddp.assert_called_once_with(model, device_ids=[1], find_unused_parameters=True)


@patch("gravitorch.distributed.auto.dist.backend", lambda *args, **kwargs: dist.Backend.NCCL)
@patch("gravitorch.distributed.auto.dist.get_local_rank", lambda *args, **kwargs: 1)
def test_wrap_distributed_data_parallel_nccl_backend_sync_batch_norm() -> None:
    with patch("gravitorch.distributed.auto.DistributedDataParallel") as mocked_ddp:
        model = nn.Linear(4, 6)
        _wrap_distributed_data_parallel(model, sync_batch_norm=True)
        mocked_ddp.assert_called_once_with(model, device_ids=[1])
