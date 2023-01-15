from collections.abc import Sequence
from typing import Union
from unittest.mock import Mock

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor

from gravitorch.utils.tensor import (
    has_name,
    partial_transpose_dict,
    permute,
    shapes_are_equal,
    str_full_tensor,
    to_tensor,
)

#####################################
#     Tests for str_full_tensor     #
#####################################


def test_str_full_tensor():
    assert str_full_tensor(torch.ones(1001)).startswith(
        "tensor([1., 1., 1., 1., 1., 1., 1., 1., 1.,"
    )


##############################
#     Tests for has_name     #
##############################


def test_has_name_true_partial_names():
    assert has_name(torch.ones(2, 3, names=("B", None)))


def test_has_name_true_full_names():
    assert has_name(torch.ones(2, 3, names=("B", "F")))


def test_has_name_false():
    assert not has_name(torch.ones(2, 3))


#############################
#     Tests for permute     #
#############################


def test_permute_1d():
    assert permute(tensor=torch.arange(4), permutation=torch.tensor([0, 2, 1, 3])).equal(
        torch.tensor([0, 2, 1, 3])
    )


def test_permute_2d_dim_0():
    assert permute(
        tensor=torch.arange(20).view(4, 5), permutation=torch.tensor([0, 2, 1, 3])
    ).equal(
        torch.tensor([[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]])
    )


def test_permute_2d_dim_1():
    assert permute(
        tensor=torch.arange(20).view(4, 5), permutation=torch.tensor([0, 4, 2, 1, 3]), dim=1
    ).equal(
        torch.tensor([[0, 4, 2, 1, 3], [5, 9, 7, 6, 8], [10, 14, 12, 11, 13], [15, 19, 17, 16, 18]])
    )


def test_permute_3d_dim_2():
    assert permute(
        tensor=torch.arange(20).view(2, 2, 5), permutation=torch.tensor([0, 4, 2, 1, 3]), dim=2
    ).equal(
        torch.tensor(
            [[[0, 4, 2, 1, 3], [5, 9, 7, 6, 8]], [[10, 14, 12, 11, 13], [15, 19, 17, 16, 18]]]
        )
    )


############################################
#     Tests for partial_transpose_dict     #
############################################


def test_partial_transpose_dict_empty_config():
    x = {"my_key": torch.arange(10).view(2, 5)}
    y = partial_transpose_dict(x, {})
    assert objects_are_equal(x, y)
    assert x["my_key"] is y["my_key"]


@mark.parametrize("dims", ((0, 1), [0, 1], (1, 0)))
def test_partial_transpose_dict_2d(dims: Sequence[int]):
    assert objects_are_equal(
        partial_transpose_dict({"my_key": torch.arange(10).view(2, 5)}, {"my_key": dims}),
        {"my_key": torch.tensor([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])},
    )


def test_partial_transpose_dict_3d_transpose_0_1():
    assert objects_are_equal(
        partial_transpose_dict({"my_key": torch.arange(24).view(2, 3, 4)}, {"my_key": (0, 1)}),
        {
            "my_key": torch.tensor(
                [
                    [[0, 1, 2, 3], [12, 13, 14, 15]],
                    [[4, 5, 6, 7], [16, 17, 18, 19]],
                    [[8, 9, 10, 11], [20, 21, 22, 23]],
                ]
            )
        },
    )


def test_partial_transpose_dict_3d_transpose_0_2():
    assert objects_are_equal(
        partial_transpose_dict({"my_key": torch.arange(24).view(2, 3, 4)}, {"my_key": (0, 2)}),
        {
            "my_key": torch.tensor(
                [
                    [[0, 12], [4, 16], [8, 20]],
                    [[1, 13], [5, 17], [9, 21]],
                    [[2, 14], [6, 18], [10, 22]],
                    [[3, 15], [7, 19], [11, 23]],
                ]
            )
        },
    )


def test_partial_transpose_dict_3d_transpose_1_2():
    assert objects_are_equal(
        partial_transpose_dict({"my_key": torch.arange(24).view(2, 3, 4)}, {"my_key": (1, 2)}),
        {
            "my_key": torch.tensor(
                [
                    [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
                    [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]],
                ]
            )
        },
    )


def test_partial_transpose_dict_missing_key():
    with raises(ValueError):
        partial_transpose_dict({"my_key": torch.arange(10).view(2, 5)}, {"another_key": [0, 1]})


###############################
#     Tests for to_tensor     #
###############################


@mark.parametrize(
    "value",
    (
        torch.tensor([-3, 1, 7]),
        [-3, 1, 7],
        (-3, 1, 7),
        np.array([-3, 1, 7]),
    ),
)
def test_to_tensor(value: Union[Tensor, Sequence, np.ndarray]):
    assert to_tensor(value).equal(torch.tensor([-3, 1, 7]))


def test_to_tensor_int():
    assert to_tensor(1).equal(torch.tensor(1, dtype=torch.long))


def test_to_tensor_float():
    assert to_tensor(1.5).equal(torch.tensor(1.5, dtype=torch.float))


def test_to_tensor_empty_list():
    assert to_tensor([]).equal(torch.tensor([]))


def test_to_tensor_incorrect():
    with raises(TypeError):
        to_tensor(Mock())


######################################
#     Tests for shapes_are_equal     #
######################################


def test_shapes_are_equal_0_tensor():
    assert not shapes_are_equal([])


def test_shapes_are_equal_1_tensor():
    assert shapes_are_equal([torch.rand(2, 3)])


@mark.parametrize("shape", ((4,), (2, 3), (2, 3, 4)))
def test_shapes_are_equal_true_2_tensors(shape: tuple[int, ...]):
    assert shapes_are_equal([torch.rand(*shape), torch.rand(*shape)])


def test_shapes_are_equal_true_3_tensors():
    assert shapes_are_equal([torch.rand(2, 3), torch.zeros(2, 3), torch.ones(2, 3)])


def test_shapes_are_equal_false_2_tensors():
    assert not shapes_are_equal([torch.rand(2, 3), torch.rand(2, 3, 1)])


def test_shapes_are_equal_false_3_tensors():
    assert not shapes_are_equal([torch.rand(2, 3), torch.zeros(2, 3, 4), torch.ones(2)])
