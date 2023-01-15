from unittest.mock import patch

import torch

from gravitorch.utils.tensor import LazyFlattedTensor

#######################################
#     Tests for LazyFlattedTensor     #
#######################################


def test_lazy_flatted_tensor_init_values_none():
    lazy_tensor = LazyFlattedTensor()
    assert lazy_tensor._values.equal(torch.tensor([]))
    assert not lazy_tensor._buffer


def test_lazy_flatted_tensor_init_values_tensor():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    assert lazy_tensor._values.equal(torch.arange(4))
    assert not lazy_tensor._buffer


def test_lazy_flatted_tensor_str():
    assert str(LazyFlattedTensor()).startswith("LazyFlattedTensor(")


@patch("gravitorch.utils.tensor.flatted.all_gather_tensor_varshape", lambda tensor: [tensor])
def test_lazy_flatted_tensor_all_reduce_non_distributed():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    lazy_tensor_reduced = lazy_tensor.all_reduce()
    assert lazy_tensor is not lazy_tensor_reduced
    assert lazy_tensor.equal(
        LazyFlattedTensor(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    )
    assert lazy_tensor_reduced.equal(
        LazyFlattedTensor(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    )


@patch(
    "gravitorch.utils.tensor.flatted.all_gather_tensor_varshape",
    lambda tensor: [tensor, torch.tensor([3, 2, 1])],
)
def test_lazy_flatted_tensor_all_reduce_distributed():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    lazy_tensor_reduced = lazy_tensor.all_reduce()
    assert lazy_tensor is not lazy_tensor_reduced
    assert lazy_tensor.equal(
        LazyFlattedTensor(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    )
    assert lazy_tensor_reduced.equal(
        LazyFlattedTensor(torch.tensor([0, 1, 2, 3, -3, 1, 7, 3, 2, 1], dtype=torch.long))
    )


def test_lazy_flatted_tensor_all_reduce_empty():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor_reduced = lazy_tensor.all_reduce()
    assert lazy_tensor is not lazy_tensor_reduced
    assert lazy_tensor.equal(LazyFlattedTensor())
    assert lazy_tensor_reduced.equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_clear():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    lazy_tensor.clear()
    assert lazy_tensor.equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_clear_empty():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.clear()
    assert lazy_tensor.equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_clone_values_without_buffer():
    lazy_tensor = LazyFlattedTensor(torch.arange(6))
    lazy_tensor_cloned = lazy_tensor.clone()
    lazy_tensor.values().add_(1)
    assert lazy_tensor is not lazy_tensor_cloned
    assert lazy_tensor.equal(LazyFlattedTensor(torch.arange(6).add(1)))
    assert lazy_tensor_cloned.equal(LazyFlattedTensor(torch.arange(6)))


def test_lazy_flatted_tensor_clone_values_with_buffer():
    lazy_tensor = LazyFlattedTensor(torch.arange(6))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    lazy_tensor_cloned = lazy_tensor.clone()
    lazy_tensor.values().add_(1)
    assert lazy_tensor is not lazy_tensor_cloned
    assert lazy_tensor.equal(LazyFlattedTensor(torch.tensor([1, 2, 3, 4, 5, 6, -2, 2, 8])))
    assert lazy_tensor_cloned.equal(LazyFlattedTensor(torch.tensor([0, 1, 2, 3, 4, 5, -3, 1, 7])))


def test_lazy_flatted_tensor_clone_empty():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor_cloned = lazy_tensor.clone()
    assert lazy_tensor is not lazy_tensor_cloned
    assert lazy_tensor.equal(LazyFlattedTensor())
    assert lazy_tensor_cloned.equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_consolidate():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    lazy_tensor.consolidate()
    assert lazy_tensor.equal(
        LazyFlattedTensor(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    )


def test_lazy_flatted_tensor_consolidate_empty_buffer():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.consolidate()
    assert lazy_tensor.equal(LazyFlattedTensor(torch.tensor([0, 1, 2, 3], dtype=torch.long)))


def test_lazy_flatted_tensor_consolidate_empty():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.consolidate()
    assert lazy_tensor.equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_equal_true_values_without_buffer():
    assert LazyFlattedTensor(torch.arange(6)).equal(LazyFlattedTensor(torch.arange(6)))


def test_lazy_flatted_tensor_equal_true_values_with_buffer():
    tensor1 = LazyFlattedTensor(torch.arange(6))
    tensor1.update(torch.tensor([-1.0, 4.0]))
    tensor2 = LazyFlattedTensor(torch.arange(6))
    tensor2.update(torch.tensor([-1.0, 4.0]))
    assert tensor1.equal(tensor2)


def test_lazy_flatted_tensor_equal_true_empty():
    assert LazyFlattedTensor().equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_equal_false_different_type():
    assert not LazyFlattedTensor().equal(torch.arange(6))


def test_lazy_flatted_tensor_equal_false_self_empty():
    assert not LazyFlattedTensor().equal(LazyFlattedTensor(torch.arange(6)))


def test_lazy_flatted_tensor_equal_false_other_empty():
    assert not LazyFlattedTensor(torch.arange(6)).equal(LazyFlattedTensor())


def test_lazy_flatted_tensor_equal_false_same_values_different_buffers():
    tensor1 = LazyFlattedTensor(torch.arange(6))
    tensor1.update(torch.tensor([-1.0, 4.0]))
    tensor2 = LazyFlattedTensor(torch.arange(6))
    tensor2.update(torch.tensor([-2.0, 4.0]))
    assert not tensor1.equal(tensor2)


def test_lazy_flatted_tensor_numel_empty():
    assert LazyFlattedTensor().numel() == 0


def test_lazy_flatted_tensor_numel_without_buffer():
    assert LazyFlattedTensor(torch.arange(6)).numel() == 6


def test_lazy_flatted_tensor_numel_with_buffer():
    lazy_tensor = LazyFlattedTensor(torch.arange(6))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    assert lazy_tensor.numel() == 9


def test_lazy_flatted_tensor_update_1d_tensor():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.update(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    assert lazy_tensor.values().equal(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))


def test_lazy_flatted_tensor_update_2d_tensor():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.update(torch.arange(6, dtype=torch.float).view(2, 3))
    assert lazy_tensor.values().equal(
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float)
    )


def test_lazy_flatted_tensor_update_float_tensor():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.update(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))
    assert lazy_tensor.values().equal(torch.tensor([-3.0, 1.0, 7.0], dtype=torch.float))


def test_lazy_flatted_tensor_update_long_tensor():
    lazy_tensor = LazyFlattedTensor()
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    assert lazy_tensor.values().equal(torch.tensor([-3, 1, 7], dtype=torch.long))


def test_lazy_flatted_tensor_values():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    assert lazy_tensor.values().equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))


def test_lazy_flatted_tensor_values_empty():
    assert LazyFlattedTensor().values().equal(torch.tensor([]))


def test_lazy_flatted_tensor_values_duplicate_call():
    lazy_tensor = LazyFlattedTensor(values=torch.arange(4))
    lazy_tensor.update(torch.tensor([-3, 1, 7]))
    values1 = lazy_tensor.values()
    assert values1.equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    values2 = lazy_tensor.values()
    assert values2.equal(torch.tensor([0, 1, 2, 3, -3, 1, 7], dtype=torch.long))
    assert values1 is values2
