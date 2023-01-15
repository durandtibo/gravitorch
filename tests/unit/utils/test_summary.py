import numpy as np
import torch

from gravitorch.utils.summary import concise_summary

#####################################
#     Tests for concise_summary     #
#####################################


def test_concise_summary_torch_tensor_float():
    assert concise_summary(torch.ones(2, 3)) == (
        "<class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.float32 | mean=1.000000 | "
        "std=0.000000 | min=1.000000 | max=1.000000"
    )


def test_concise_summary_torch_tensor_long():
    assert concise_summary(torch.ones(2, 3, dtype=torch.long)) == (
        "<class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.int64 | mean=1.000000 | "
        "std=0.000000 | min=1 | max=1"
    )


def test_concise_summary_torch_tensor_bool():
    assert concise_summary(torch.ones(2, 3, dtype=torch.bool)) == (
        "<class 'torch.Tensor'> | shape=torch.Size([2, 3]) | dtype=torch.bool"
    )


def test_concise_summary_numpy_array():
    assert concise_summary(np.ones((2, 3))).startswith(
        "<class 'numpy.ndarray'> | shape=(2, 3) | dtype=float64 | mean=1.000000 | std=0.000000 | "
        "min=1.000000 | max=1.000000"
    )


def test_concise_summary_list():
    assert concise_summary([1, 2, 3]) == (
        "<class 'list'> | length=3\n"
        "  (0) <class 'int'> | value=1\n"
        "  (1) <class 'int'> | value=2\n"
        "  (2) <class 'int'> | value=3"
    )


def test_concise_summary_list_length_3_max_length_3():
    assert concise_summary([i for i in range(3)], max_length=3) == (
        "<class 'list'> | length=3\n"
        "  (0) <class 'int'> | value=0\n"
        "  (1) <class 'int'> | value=1\n"
        "  (2) <class 'int'> | value=2"
    )


def test_concise_summary_list_length_10_max_length_3():
    assert concise_summary([i for i in range(10)], max_length=3) == (
        "<class 'list'> | length=10\n"
        "  (0) <class 'int'> | value=0\n"
        "  (1) <class 'int'> | value=1\n"
        "  (2) <class 'int'> | value=2\n"
        "  ..."
    )


def test_concise_summary_list_length_10_max_length_5():
    assert concise_summary([i for i in range(10)], max_length=5) == (
        "<class 'list'> | length=10\n"
        "  (0) <class 'int'> | value=0\n"
        "  (1) <class 'int'> | value=1\n"
        "  (2) <class 'int'> | value=2\n"
        "  (3) <class 'int'> | value=3\n"
        "  (4) <class 'int'> | value=4\n"
        "  ..."
    )


def test_concise_summary_tuple():
    assert concise_summary((1, 2, 3)) == (
        "<class 'tuple'> | length=3\n"
        "  (0) <class 'int'> | value=1\n"
        "  (1) <class 'int'> | value=2\n"
        "  (2) <class 'int'> | value=3"
    )


def test_concise_summary_mapping():
    assert concise_summary({"key1": 1, "key2": 2, "key3": 3}) == (
        "<class 'dict'> | length=3\n"
        "  (key1) <class 'int'> | value=1\n"
        "  (key2) <class 'int'> | value=2\n"
        "  (key3) <class 'int'> | value=3"
    )


def test_concise_summary_mapping_length_3_max_length_3():
    assert concise_summary({f"key{i}": i for i in range(3)}, max_length=3) == (
        "<class 'dict'> | length=3\n"
        "  (key0) <class 'int'> | value=0\n"
        "  (key1) <class 'int'> | value=1\n"
        "  (key2) <class 'int'> | value=2"
    )


def test_concise_summary_mapping_length_10_max_length_3():
    assert concise_summary({f"key{i}": i for i in range(10)}, max_length=3) == (
        "<class 'dict'> | length=10\n"
        "  (key0) <class 'int'> | value=0\n"
        "  (key1) <class 'int'> | value=1\n"
        "  (key2) <class 'int'> | value=2\n"
        "  ..."
    )


def test_concise_summary_mapping_length_10_max_length_5():
    assert concise_summary({f"key{i}": i for i in range(10)}, max_length=5) == (
        "<class 'dict'> | length=10\n"
        "  (key0) <class 'int'> | value=0\n"
        "  (key1) <class 'int'> | value=1\n"
        "  (key2) <class 'int'> | value=2\n"
        "  (key3) <class 'int'> | value=3\n"
        "  (key4) <class 'int'> | value=4\n"
        "  ..."
    )


def test_concise_summary_int():
    assert concise_summary(35) == "<class 'int'> | value=35"


def test_concise_summary_float():
    assert concise_summary(1.5) == "<class 'float'> | value=1.5"


def test_concise_summary_str():
    assert concise_summary("abc") == "<class 'str'> | value=abc"
