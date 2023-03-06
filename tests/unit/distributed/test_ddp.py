from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark, raises

from gravitorch.distributed import ddp
from gravitorch.utils import get_available_devices

###########################################
#     Tests for broadcast_object_list     #
###########################################


@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: False)
def test_broadcast_object_list_not_distributed() -> None:
    x = [35]
    ddp.broadcast_object_list(x)
    assert x == [35]


@mark.parametrize("object_list", ([1], [35, 42]))
@mark.parametrize("src", (0, 1))
@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: True)
def test_broadcast_object_list_distributed(object_list: list, src: int, device: str) -> None:
    device = torch.device(device)
    with patch("gravitorch.distributed.ddp.tdist.broadcast_object_list") as broadcast_mock:
        ddp.broadcast_object_list(object_list=object_list, src=src, device=device)
        broadcast_mock.assert_called_once_with(object_list=object_list, src=src, device=device)


#################################
#     Tests for sync_reduce     #
#################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("is_distributed", (True, False))
def test_sync_reduce_sum_number(device: str, is_distributed: bool) -> None:
    with patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: is_distributed):
        assert ddp.sync_reduce(35, ddp.SUM) == 35


@mark.parametrize("device", get_available_devices())
@mark.parametrize("is_distributed", (True, False))
def test_sync_reduce_sum_tensor(device: str, is_distributed: bool) -> None:
    with patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: is_distributed):
        var_reduced = ddp.sync_reduce(torch.ones(2, 3, device=device), ddp.SUM)
        assert var_reduced.equal(torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: True)
def test_sync_reduce_avg_number_world_size_2_is_distributed(device: str) -> None:
    assert ddp.sync_reduce(8, ddp.AVG) == 4


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: False)
def test_sync_reduce_avg_number_world_size_2_is_not_distributed(device: str) -> None:
    assert ddp.sync_reduce(8, ddp.AVG) == 8


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: True)
def test_sync_reduce_avg_tensor_world_size_2_is_distributed(device: str) -> None:
    var_reduced = ddp.sync_reduce(torch.ones(2, 3, device=device), ddp.AVG)
    assert var_reduced.equal(0.5 * torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: False)
def test_sync_reduce_avg_tensor_world_size_2_is_not_distributed(device: str) -> None:
    x = torch.ones(2, 3, device=device)
    x_reduced = ddp.sync_reduce(x, ddp.AVG)
    assert x_reduced.equal(x)  # no-op because not distributed


##################################
#     Tests for sync_reduce_     #
##################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("is_distributed", (True, False))
def test_sync_reduce__sum(device: str, is_distributed: bool) -> None:
    with patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: is_distributed):
        variable = torch.ones(2, 3, device=device)
        ddp.sync_reduce_(variable, ddp.SUM)
        assert variable.equal(torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: True)
def test_sync_reduce__avg_world_size_2_is_distributed(device: str) -> None:
    variable = torch.ones(2, 3, device=device)
    ddp.sync_reduce_(variable, ddp.AVG)
    assert variable.equal(0.5 * torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
@patch("gravitorch.distributed.ddp.get_world_size", lambda *args, **kwargs: 2)
@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: False)
def test_sync_reduce__avg_world_size_2_is_not_distributed(device: str) -> None:
    variable = torch.ones(2, 3, device=device)
    ddp.sync_reduce_(variable, ddp.AVG)
    assert variable.equal(torch.ones(2, 3, device=device))  # no-op because not distributed


def test_sync_reduce__incorrect_input() -> None:
    with raises(TypeError):
        ddp.sync_reduce_(1, ddp.SUM)


################################################
#     Tests for all_gather_tensor_varshape     #
################################################


@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: False)
def test_all_gather_tensor_varshape_not_distributed() -> None:
    assert objects_are_equal(ddp.all_gather_tensor_varshape(torch.arange(6)), [torch.arange(6)])


@patch("gravitorch.distributed.ddp.is_distributed", lambda *args, **kwargs: True)
@patch("gravitorch.distributed.ddp.all_gather", lambda tensor: tensor)
@patch("gravitorch.distributed.ddp.dist_device", lambda *args: torch.device("cpu"))
def test_all_gather_tensor_varshape_distributed() -> None:
    assert objects_are_equal(ddp.all_gather_tensor_varshape(torch.arange(6)), [torch.arange(6)])
