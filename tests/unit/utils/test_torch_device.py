from collections import OrderedDict
from collections.abc import Mapping
from unittest.mock import patch

import torch
from coola import objects_are_equal
from pytest import mark
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from gravitorch.utils.torch_device import get_available_devices, move_to_device

###########################################
#     Tests for get_available_devices     #
###########################################


@patch("torch.cuda.is_available", lambda *args, **kwargs: False)
def test_get_available_devices_cpu():
    assert get_available_devices() == ("cpu",)


@patch("torch.cuda.is_available", lambda *args, **kwargs: True)
@patch("torch.cuda.device_count", lambda *args, **kwargs: 1)
def test_get_available_devices_cpu_and_gpu():
    assert get_available_devices() == ("cpu", "cuda:0")


####################################
#     Tests for send_to_device     #
####################################


@mark.parametrize("device", get_available_devices())
def test_send_to_device_tensor(device: str):
    device = torch.device(device)
    obj = move_to_device(torch.ones(2, 3), device)
    assert torch.is_tensor(obj)
    assert obj.equal(torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
def test_send_to_device_packed_sequence(device: str):
    device = torch.device(device)
    obj = move_to_device(pack_sequence([torch.ones(2, 4, device=device) for _ in range(2)]), device)
    assert isinstance(obj, PackedSequence)
    assert obj.data.device.type == device.type


@mark.parametrize("device", get_available_devices())
def test_send_to_device_nn_module(device: str):
    device = torch.device(device)
    obj = move_to_device(nn.Linear(4, 6), device)
    assert isinstance(obj, nn.Module)
    assert obj.weight.data.device.type == device.type


@mark.parametrize("device", get_available_devices())
def test_send_to_device_list(device: str):
    device = torch.device(device)
    obj = move_to_device([torch.ones(2, 3), torch.ones(4)], device)
    assert isinstance(obj, list)
    assert objects_are_equal(obj, [torch.ones(2, 3, device=device), torch.ones(4, device=device)])


@mark.parametrize("device", get_available_devices())
def test_send_to_device_tuple(device: str):
    device = torch.device(device)
    obj = move_to_device((torch.ones(2, 3), torch.ones(4)), device)
    assert isinstance(obj, tuple)
    assert objects_are_equal(obj, (torch.ones(2, 3, device=device), torch.ones(4, device=device)))


@mark.parametrize("device", get_available_devices())
def test_send_to_device_set(device: str):
    device = torch.device(device)
    obj = move_to_device({torch.ones(2, 3), torch.ones(2, 3)}, device)
    assert isinstance(obj, set)
    for value in obj:
        assert value.device.type == device.type
        assert value.equal(torch.ones(2, 3, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize(
    "obj,obj_cls",
    (
        ({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}, dict),
        (OrderedDict({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}), OrderedDict),
    ),
)
def test_send_to_device_dict(device: str, obj: Mapping, obj_cls: type[object]):
    print(obj, obj_cls)
    device = torch.device(device)
    obj = move_to_device(obj, device)
    assert isinstance(obj, obj_cls)
    assert obj["tensor1"].device.type == device.type
    assert obj["tensor1"].equal(torch.ones(2, 3, device=device))
    assert obj["tensor2"].device.type == device.type
    assert obj["tensor2"].equal(torch.ones(4, device=device))


@mark.parametrize("device", get_available_devices())
def test_send_to_device_dict_nested(device: str):
    device = torch.device(device)
    obj = move_to_device({"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)}, device)
    assert isinstance(obj, dict)
    assert objects_are_equal(
        obj, {"list": [1, torch.zeros(2, 3, device=device)], "tensor": torch.ones(4, device=device)}
    )
