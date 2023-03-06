import numpy as np
import torch
from coola import objects_are_equal

from gravitorch.experimental.object_ops import add_objects, sub_objects

#################################
#     Tests for add_objects     #
#################################


def test_add_objects_integer() -> None:
    assert add_objects(1, 2) == 3


def test_add_objects_float() -> None:
    assert add_objects(1.5, 2.5) == 4.0


def test_add_objects_torch_tensor() -> None:
    assert add_objects(torch.ones(2, 3), 2 * torch.ones(2, 3)).equal(3 * torch.ones(2, 3))


def test_add_objects_numpy_array() -> None:
    assert np.array_equal(add_objects(np.ones((2, 3)), 2 * np.ones((2, 3))), 3 * np.ones((2, 3)))


def test_add_objects_empty_list() -> None:
    assert add_objects([], []) == []


def test_add_objects_list() -> None:
    assert add_objects([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def test_add_objects_list_torch_tensor() -> None:
    assert objects_are_equal(
        add_objects([torch.ones(2, 3), torch.ones(3)], [torch.ones(2, 3), 2 * torch.ones(3)]),
        [2 * torch.ones(2, 3), 3 * torch.ones(3)],
    )


def test_add_objects_empty_tuple() -> None:
    assert add_objects((), ()) == ()


def test_add_objects_tuple() -> None:
    assert add_objects((1, 2, 3), (4, 5, 6)) == (5, 7, 9)


def test_add_objects_tuple_torch_tensor() -> None:
    assert objects_are_equal(
        add_objects((torch.ones(2, 3), torch.ones(3)), (torch.ones(2, 3), 2 * torch.ones(3))),
        (2 * torch.ones(2, 3), 3 * torch.ones(3)),
    )


def test_add_objects_hybrid_list_tuple() -> None:
    assert add_objects([1, 2, 3], (4, 5, 6)) == [5, 7, 9]


def test_add_objects_empty_dict() -> None:
    assert add_objects({}, {}) == {}


def test_add_objects_dict() -> None:
    assert add_objects({"abc": 1, "def": 2}, {"abc": -1, "def": 2}) == {"abc": 0, "def": 4}


def test_add_objects_dict_torch_tensor() -> None:
    assert objects_are_equal(
        add_objects(
            {"abc": torch.ones(2, 3), "def": torch.ones(3)},
            {"abc": torch.ones(2, 3), "def": 2 * torch.ones(3)},
        ),
        {"abc": 2 * torch.ones(2, 3), "def": 3 * torch.ones(3)},
    )


#################################
#     Tests for sub_objects     #
#################################


def test_sub_objects_integer() -> None:
    assert sub_objects(2, 1) == 1


def test_sub_objects_float() -> None:
    assert sub_objects(1.5, 2.5) == -1.0


def test_sub_objects_torch_tensor() -> None:
    assert sub_objects(4 * torch.ones(2, 3), torch.ones(2, 3)).equal(3 * torch.ones(2, 3))


def test_sub_objects_numpy_array() -> None:
    assert np.array_equal(sub_objects(4 * np.ones((2, 3)), np.ones((2, 3))), 3 * np.ones((2, 3)))


def test_sub_objects_empty_list() -> None:
    assert sub_objects([], []) == []


def test_sub_objects_list() -> None:
    assert sub_objects([1, 2, 3], [2, 4, 6]) == [-1, -2, -3]


def test_sub_objects_list_torch_tensor() -> None:
    assert objects_are_equal(
        sub_objects([4 * torch.ones(2, 3), torch.ones(3)], [torch.ones(2, 3), 2 * torch.ones(3)]),
        [3 * torch.ones(2, 3), -torch.ones(3)],
    )


def test_sub_objects_empty_tuple() -> None:
    assert sub_objects((), ()) == ()


def test_sub_objects_tuple() -> None:
    assert sub_objects((1, 2, 3), (2, 4, 6)) == (-1, -2, -3)


def test_sub_objects_tuple_torch_tensor() -> None:
    assert objects_are_equal(
        sub_objects((4 * torch.ones(2, 3), torch.ones(3)), (torch.ones(2, 3), 2 * torch.ones(3))),
        (3 * torch.ones(2, 3), -torch.ones(3)),
    )


def test_sub_objects_hybrid_list_tuple() -> None:
    assert sub_objects([1, 2, 3], (2, 4, 6)) == [-1, -2, -3]


def test_sub_objects_empty_dict() -> None:
    assert sub_objects({}, {}) == {}


def test_sub_objects_dict() -> None:
    assert sub_objects({"abc": 1, "def": 2}, {"abc": -1, "def": 2}) == {"abc": 2, "def": 0}


def test_sub_objects_dict_torch_tensor() -> None:
    assert objects_are_equal(
        sub_objects(
            {"abc": 4 * torch.ones(2, 3), "def": torch.ones(3)},
            {"abc": torch.ones(2, 3), "def": 2 * torch.ones(3)},
        ),
        {"abc": 3 * torch.ones(2, 3), "def": -torch.ones(3)},
    )
