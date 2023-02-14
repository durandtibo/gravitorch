from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark

from gravitorch.utils.tensor import (
    UNKNOWN,
    get_dtype,
    get_shape,
    recursive_apply,
    recursive_contiguous,
    recursive_detach,
    recursive_from_numpy,
    recursive_transpose,
)

###############################
#     Tests for get_dtype     #
###############################


def test_get_dtype_float_tensor():
    assert get_dtype(torch.ones(2, 3, dtype=torch.float)) == torch.float


def test_get_dtype_long_tensor():
    assert get_dtype(torch.ones(2, 3, dtype=torch.long)) == torch.long


def test_get_dtype_dict():
    assert get_dtype({"key1": torch.ones(2, 3), "key2": "abc"}) == {
        "key1": torch.float,
        "key2": UNKNOWN,
    }


def test_get_dtype_ordered_dict():
    assert get_dtype(OrderedDict({"key1": torch.ones(2, 3), "key2": "abc"})) == OrderedDict(
        {
            "key1": torch.float,
            "key2": UNKNOWN,
        }
    )


def test_get_dtype_list():
    assert get_dtype([torch.ones(2, 3), "abc"]) == [torch.float, UNKNOWN]


def test_get_dtype_tuple():
    assert get_dtype((torch.ones(2, 3), "abc")) == (torch.float, UNKNOWN)


def test_get_dtype_not_a_tensor():
    assert get_dtype("abc") == UNKNOWN


###############################
#     Tests for get_shape     #
###############################


def test_get_shape_tensor_2d():
    assert get_shape(torch.ones(2, 3)) == (2, 3)


def test_get_shape_tensor_3d():
    assert get_shape(torch.ones(1, 2, 3)) == (1, 2, 3)


def test_get_shape_dict():
    assert get_shape({"key1": torch.ones(1, 2, 3), "key2": "abc"}) == {
        "key1": (1, 2, 3),
        "key2": UNKNOWN,
    }


def test_get_shape_ordered_dict():
    assert get_shape(OrderedDict({"key1": torch.ones(1, 2, 3), "key2": "abc"})) == OrderedDict(
        {
            "key1": (1, 2, 3),
            "key2": UNKNOWN,
        }
    )


def test_get_shape_list():
    assert get_shape([torch.ones(1, 2, 3), "abc"]) == [(1, 2, 3), UNKNOWN]


def test_get_shape_tuple():
    assert get_shape((torch.ones(1, 2, 3), "abc")) == ((1, 2, 3), UNKNOWN)


def test_get_shape_not_a_tensor():
    assert get_shape("abc") == UNKNOWN


#####################################
#     Tests for recursive_apply     #
#####################################


def test_recursive_apply_torch_tensor():
    assert recursive_apply(torch.ones(2, 3), lambda tensor: tensor.sum().item()) == 6


def test_recursive_apply_list():
    assert recursive_apply(
        [torch.ones(3, 2), torch.ones(4)], lambda tensor: tensor.sum().item()
    ) == [
        6,
        4,
    ]


def test_recursive_apply_tuple():
    assert recursive_apply(
        (torch.ones(3, 2), torch.ones(4)), lambda tensor: tensor.sum().item()
    ) == (
        6,
        4,
    )


def test_recursive_apply_set():
    assert recursive_apply(
        {torch.ones(3, 2), torch.ones(4)}, lambda tensor: tensor.sum().item()
    ) == {6, 4}


def test_recursive_apply_dict():
    assert recursive_apply(
        {"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)},
        lambda tensor: tensor.sum().item(),
    ) == {"tensor1": 6, "tensor2": 4}


def test_recursive_apply_ordered_dict():
    assert recursive_apply(
        OrderedDict({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}),
        lambda tensor: tensor.sum().item(),
    ) == OrderedDict({"tensor1": 6, "tensor2": 4})


def test_recursive_apply_dict_nested():
    assert recursive_apply(
        {"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)},
        lambda tensor: tensor.sum().item(),
    ) == {"list": [1, 0], "tensor": 4}


def test_recursive_apply_dict_nested_other_fn():
    assert recursive_apply(
        {"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)},
        tensor_fn=lambda tensor: tensor.sum().item(),
        other_fn=lambda value: UNKNOWN,
    ) == {"list": [UNKNOWN, 0], "tensor": 4}


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_apply_other_types(obj: Any):
    assert recursive_apply(obj, lambda tensor: tensor.sum().item()) == obj


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_apply_other_types_with_other_fn(obj: Any):
    assert (
        recursive_apply(obj, tensor_fn=lambda tensor: tensor, other_fn=lambda value: UNKNOWN)
        == UNKNOWN
    )


##########################################
#     Tests for recursive_contiguous     #
##########################################


def test_recursive_contiguous_torch_tensor():
    x = torch.ones(3, 2).transpose(0, 1)
    assert not x.is_contiguous()
    obj = recursive_contiguous(x)
    assert obj.equal(torch.ones(2, 3))
    assert obj.is_contiguous()


def test_recursive_contiguous_list():
    obj = recursive_contiguous([torch.ones(3, 2).transpose(0, 1), torch.ones(4)])
    assert isinstance(obj, list)
    assert obj[0].is_contiguous()
    assert obj[0].equal(torch.ones(2, 3))
    assert obj[1].is_contiguous()
    assert obj[1].equal(torch.ones(4))


def test_recursive_contiguous_tuple():
    obj = recursive_contiguous((torch.ones(3, 2).transpose(0, 1), torch.ones(4)))
    assert isinstance(obj, tuple)
    assert obj[0].is_contiguous()
    assert obj[0].equal(torch.ones(2, 3))
    assert obj[1].is_contiguous()
    assert obj[1].equal(torch.ones(4))


def test_recursive_contiguous_set():
    obj = recursive_contiguous({torch.ones(3, 2).transpose(0, 1) for _ in range(3)})
    assert isinstance(obj, set)
    assert len(obj) == 3
    for value in obj:
        assert value.is_contiguous()
        assert value.equal(torch.ones(2, 3))


@mark.parametrize(
    "obj,obj_cls",
    (
        ({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}, dict),
        (OrderedDict({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}), OrderedDict),
    ),
)
def test_recursive_contiguous_dict(obj: dict, obj_cls: type[object]):
    obj = recursive_contiguous(obj)
    assert isinstance(obj, obj_cls)
    assert obj["tensor1"].is_contiguous()
    assert obj["tensor1"].equal(torch.ones(2, 3))
    assert obj["tensor2"].is_contiguous()
    assert obj["tensor2"].equal(torch.ones(4))


def test_recursive_contiguous_dict_nested():
    obj = recursive_contiguous({"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)})
    assert isinstance(obj, dict)
    assert objects_are_equal(obj, {"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)})
    assert obj["list"][1].is_contiguous()
    assert obj["tensor"].is_contiguous()


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_contiguous_other_types(obj: Any):
    assert recursive_contiguous(obj) == obj


def test_recursive_contiguous_memory_format():
    obj = recursive_contiguous(
        {"list": [torch.zeros(2, 3, 4, 5)], "tensor": torch.ones(2, 3, 4, 5)},
        memory_format=torch.channels_last,
    )
    assert objects_are_equal(
        obj, {"list": [torch.zeros(2, 3, 4, 5)], "tensor": torch.ones(2, 3, 4, 5)}
    )
    assert obj["list"][0].is_contiguous(memory_format=torch.channels_last)
    assert obj["tensor"].is_contiguous(memory_format=torch.channels_last)


######################################
#     Tests for recursive_detach     #
######################################


def test_recursive_detach_torch_tensor():
    obj = recursive_detach(torch.ones(2, 3, requires_grad=True))
    assert obj.equal(torch.ones(2, 3))
    assert not obj.requires_grad


def test_recursive_detach_list():
    obj = recursive_detach(
        [torch.ones(2, 3, requires_grad=True), torch.ones(4, requires_grad=True)]
    )
    assert isinstance(obj, list)
    assert not obj[0].requires_grad
    assert obj[0].equal(torch.ones(2, 3))
    assert not obj[1].requires_grad
    assert obj[1].equal(torch.ones(4))


def test_recursive_detach_tuple():
    obj = recursive_detach(
        (torch.ones(2, 3, requires_grad=True), torch.ones(4, requires_grad=True))
    )
    assert isinstance(obj, tuple)
    assert not obj[0].requires_grad
    assert obj[0].equal(torch.ones(2, 3))
    assert not obj[1].requires_grad
    assert obj[1].equal(torch.ones(4))


def test_recursive_detach_set():
    obj = recursive_detach({torch.ones(2, 3, requires_grad=True) for _ in range(3)})
    assert isinstance(obj, set)
    assert len(obj) == 3
    for value in obj:
        assert not value.requires_grad
        assert value.equal(torch.ones(2, 3))


@mark.parametrize(
    "obj,obj_cls",
    (
        ({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}, dict),
        (OrderedDict({"tensor1": torch.ones(2, 3), "tensor2": torch.ones(4)}), OrderedDict),
    ),
)
def test_recursive_detach_dict(obj: dict, obj_cls: type[object]):
    obj = recursive_detach(obj)
    assert isinstance(obj, obj_cls)
    assert not obj["tensor1"].requires_grad
    assert obj["tensor1"].equal(torch.ones(2, 3))
    assert not obj["tensor2"].requires_grad
    assert obj["tensor2"].equal(torch.ones(4))


def test_recursive_detach_dict_nested():
    obj = recursive_detach({"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)})
    assert isinstance(obj, dict)
    assert objects_are_equal(obj, {"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4)})
    assert not obj["list"][1].requires_grad
    assert not obj["tensor"].requires_grad


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_detach_other_types(obj: Any):
    assert recursive_detach(obj) == obj


#########################################
#     Tests for recursive_transpose     #
#########################################


def test_recursive_transpose_torch_tensor_2d():
    assert recursive_transpose(torch.arange(10).view(2, 5), 0, 1).equal(
        torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
    )


def test_recursive_transpose_torch_tensor_3d():
    assert recursive_transpose(torch.arange(24).view(2, 3, 4), 1, 2).equal(
        torch.tensor(
            [
                [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
                [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]],
            ]
        )
    )


def test_recursive_transpose_list():
    obj = recursive_transpose([torch.ones(3, 2), torch.ones(2, 4, 1)], 0, 1)
    assert isinstance(obj, list)
    assert obj[0].equal(torch.ones(2, 3))
    assert obj[1].equal(torch.ones(4, 2, 1))


def test_recursive_transpose_tuple():
    obj = recursive_transpose((torch.ones(3, 2), torch.ones(2, 4, 1)), 0, 1)
    assert isinstance(obj, tuple)
    assert obj[0].equal(torch.ones(2, 3))
    assert obj[1].equal(torch.ones(4, 2, 1))


def test_recursive_transpose_set():
    obj = recursive_transpose({torch.ones(3, 2) for _ in range(3)}, 0, 1)
    assert isinstance(obj, set)
    assert len(obj) == 3
    for value in obj:
        assert value.equal(torch.ones(2, 3))


@mark.parametrize(
    "obj,obj_cls",
    (
        ({"tensor1": torch.ones(3, 2), "tensor2": torch.ones(2, 4, 1)}, dict),
        (OrderedDict({"tensor1": torch.ones(3, 2), "tensor2": torch.ones(2, 4, 1)}), OrderedDict),
    ),
)
def test_recursive_transpose_dict(obj: dict, obj_cls: type[object]):
    obj = recursive_transpose(obj, 0, 1)
    assert isinstance(obj, obj_cls)
    assert obj["tensor1"].equal(torch.ones(2, 3))
    assert obj["tensor2"].equal(torch.ones(4, 2, 1))


def test_recursive_transpose_dict_nested():
    obj = recursive_transpose({"list": [1, torch.zeros(3, 2)], "tensor": torch.ones(2, 4, 1)}, 0, 1)
    assert objects_are_equal(obj, {"list": [1, torch.zeros(2, 3)], "tensor": torch.ones(4, 2, 1)})


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_transpose_other_types(obj: Any):
    assert recursive_transpose(obj, 0, 1) == obj


##########################################
#     Tests for recursive_from_numpy     #
##########################################


def test_recursive_from_numpy_tensor():
    assert recursive_from_numpy(torch.arange(5)).equal(torch.arange(5))


def test_recursive_from_numpy_ndarray():
    assert recursive_from_numpy(np.arange(5, dtype=np.int32)).equal(torch.arange(5))


def test_recursive_from_numpy_list():
    assert objects_are_equal(
        recursive_from_numpy([np.ones((3, 2), dtype=np.float32), torch.zeros((2, 4, 1))]),
        [torch.ones(3, 2), torch.zeros(2, 4, 1)],
    )


def test_recursive_from_numpy_tuple():
    assert objects_are_equal(
        recursive_from_numpy((np.ones((3, 2), dtype=np.float32), torch.zeros((2, 4, 1)))),
        (torch.ones(3, 2), torch.zeros(2, 4, 1)),
    )


def test_recursive_from_numpy_set():
    assert recursive_from_numpy({1, "abc", 2}) == {1, "abc", 2}


@mark.parametrize(
    "data,target",
    (
        (
            {"array1": np.ones((3, 2), dtype=np.float32), "array2": np.zeros((2, 4, 1))},
            {"array1": torch.ones(3, 2), "array2": torch.zeros(2, 4, 1, dtype=torch.float64)},
        ),
        (
            OrderedDict(
                {"array1": np.ones((3, 2), dtype=np.float32), "array2": np.zeros((2, 4, 1))}
            ),
            OrderedDict(
                {"array1": torch.ones(3, 2), "array2": torch.zeros(2, 4, 1, dtype=torch.float64)}
            ),
        ),
    ),
)
def test_recursive_from_numpy_dict(data: dict, target: dict):
    assert objects_are_equal(recursive_from_numpy(data), target, show_difference=True)


@mark.parametrize("obj", (1, 2.3, "abc"))
def test_recursive_from_numpy_other_types(obj: Any):
    assert recursive_from_numpy(obj) == obj
