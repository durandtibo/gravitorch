from typing import Any, Union
from unittest.mock import Mock, patch

import numpy as np
import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor

from gravitorch.datapipes.iter import SourceWrapper, TensorDictShuffler
from gravitorch.datapipes.iter.shuffling import (
    get_first_dimension,
    shuffle_tensor_mapping,
    shuffle_tensors,
)
from gravitorch.utils.seed import get_torch_generator

########################################
#     Tests for TensorDictShuffler     #
########################################


def test_tensor_dict_shuffler_str() -> None:
    assert str(TensorDictShuffler(SourceWrapper([]))).startswith("TensorDictShufflerIterDataPipe(")


@mark.parametrize("random_seed", (1, 2))
def test_tensor_dict_shuffler_iter_random_seed(random_seed: int) -> None:
    assert TensorDictShuffler(SourceWrapper([]), random_seed=random_seed).random_seed == random_seed


@patch(
    "gravitorch.datapipes.iter.shuffling.torch.randperm",
    lambda *args, **kwargs: torch.tensor([0, 2, 1, 3]),
)
def test_tensor_dict_shuffler_iter() -> None:
    source = SourceWrapper([{"key": torch.arange(4) + i} for i in range(3)])
    assert objects_are_equal(
        list(TensorDictShuffler(source)),
        [
            {"key": torch.tensor([0, 2, 1, 3])},
            {"key": torch.tensor([1, 3, 2, 4])},
            {"key": torch.tensor([2, 4, 3, 5])},
        ],
    )


@mark.parametrize("dim", (0, 1, 2))
def test_tensor_dict_shuffler_iter_dim_int(dim: int) -> None:
    source = SourceWrapper([{"key": torch.arange(4) + i} for i in range(3)])
    with patch("gravitorch.datapipes.iter.shuffling.shuffle_tensor_mapping") as shuffle_mock:
        next(iter(TensorDictShuffler(source, dim=dim)))
        assert shuffle_mock.call_args.kwargs["dim"] == dim


@mark.parametrize("dim", (0, 1, 2))
def test_tensor_dict_shuffler_iter_dim_dict(dim: int) -> None:
    source = SourceWrapper([{"key": torch.arange(4) + i} for i in range(3)])
    with patch("gravitorch.datapipes.iter.shuffling.shuffle_tensor_mapping") as shuffle_mock:
        next(iter(TensorDictShuffler(source, dim={"key": dim})))
        assert shuffle_mock.call_args.kwargs["dim"] == {"key": dim}


def test_tensor_dict_shuffler_iter_same_random_seed() -> None:
    source = SourceWrapper([{"key": torch.arange(4) + i} for i in range(3)])
    assert objects_are_equal(
        list(TensorDictShuffler(source, random_seed=1)),
        list(TensorDictShuffler(source, random_seed=1)),
    )


def test_tensor_dict_shuffler_iter_different_random_seeds() -> None:
    source = SourceWrapper([{"key": torch.arange(4) + i} for i in range(3)])
    assert not objects_are_equal(
        list(TensorDictShuffler(source, random_seed=1)),
        list(TensorDictShuffler(source, random_seed=2)),
    )


def test_tensor_dict_shuffler_len() -> None:
    assert len(TensorDictShuffler(Mock(__len__=Mock(return_value=5)))) == 5


def test_tensor_dict_shuffler_no_len() -> None:
    source = SourceWrapper({"key": torch.arange(4) + i} for i in range(3))
    with raises(TypeError):
        len(TensorDictShuffler(source))


#####################################
#     Tests for shuffle_tensors     #
#####################################


def test_shuffle_tensors_generator() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    generator = Mock(spec=torch.Generator)
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        shuffle_tensors([torch.arange(4), torch.arange(20).view(4, 5)], generator=generator)
        randperm_mock.assert_called_once_with(4, generator=generator)


def test_shuffle_tensors_generator_default() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        shuffle_tensors([torch.arange(4), torch.arange(20).view(4, 5)])
        randperm_mock.assert_called_once_with(4, generator=None)


def test_shuffle_tensors_dim_0() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensors([torch.arange(4), torch.arange(20).view(4, 5)]),
            [
                torch.tensor([0, 2, 1, 3]),
                torch.tensor(
                    [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]
                ),
            ],
        )


def test_shuffle_tensors_dim_1() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensors([torch.arange(4).view(1, 4), torch.arange(20).view(5, 4)], dim=1),
            [
                torch.tensor([[0, 2, 1, 3]]),
                torch.tensor(
                    [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15], [16, 18, 17, 19]]
                ),
            ],
        )


def test_shuffle_tensors_dim_2() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensors([torch.arange(4).view(1, 1, 4), torch.arange(20).view(5, 1, 4)], dim=2),
            [
                torch.tensor([[[0, 2, 1, 3]]]),
                torch.tensor(
                    [
                        [[0, 2, 1, 3]],
                        [[4, 6, 5, 7]],
                        [[8, 10, 9, 11]],
                        [[12, 14, 13, 15]],
                        [[16, 18, 17, 19]],
                    ]
                ),
            ],
        )


@mark.parametrize("dim", (0, 1, 2, 3))
def test_shuffle_tensors_dim(dim: int) -> None:
    out = shuffle_tensors([torch.rand(1, 2, 3, 4), torch.ones(1, 2, 3, 4)], dim=dim)
    assert len(out) == 2
    assert out[0].shape == (1, 2, 3, 4)
    assert out[1].equal(torch.ones(1, 2, 3, 4))


def test_shuffle_tensors_incorrect_shape() -> None:
    with raises(ValueError):
        shuffle_tensors([torch.rand(1), torch.ones(4)])


def test_shuffle_tensors_same_random_seed() -> None:
    batch = [torch.arange(4), torch.arange(20).view(4, 5)]
    assert objects_are_equal(
        shuffle_tensors(batch, generator=get_torch_generator(1)),
        shuffle_tensors(batch, generator=get_torch_generator(1)),
    )


def test_shuffle_tensors_different_random_seeds() -> None:
    batch = [torch.arange(4), torch.arange(20).view(4, 5)]
    assert not objects_are_equal(
        shuffle_tensors(batch, generator=get_torch_generator(1)),
        shuffle_tensors(batch, generator=get_torch_generator(2)),
    )


############################################
#     Tests for shuffle_tensor_mapping     #
############################################


def test_shuffle_tensor_mapping_generator() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    generator = Mock(spec=torch.Generator)
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        shuffle_tensor_mapping(
            mapping={"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)},
            generator=generator,
        )
        randperm_mock.assert_called_once_with(4, generator=generator)


def test_shuffle_tensor_mapping_generator_default() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        shuffle_tensor_mapping(
            mapping={"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)}
        )
        randperm_mock.assert_called_once_with(4, generator=None)


def test_shuffle_tensor_mapping_dim_int_0() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping({"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)}),
            {
                "key1": torch.tensor([0, 2, 1, 3]),
                "key2": torch.tensor(
                    [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]
                ),
            },
        )


def test_shuffle_tensor_mapping_dim_int_1() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping(
                {"key1": torch.arange(4).view(1, 4), "key2": torch.arange(20).view(5, 4)}, dim=1
            ),
            {
                "key1": torch.tensor([[0, 2, 1, 3]]),
                "key2": torch.tensor(
                    [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15], [16, 18, 17, 19]]
                ),
            },
        )


def test_shuffle_tensor_mapping_dim_int_2() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping(
                {"key1": torch.arange(4).view(1, 1, 4), "key2": torch.arange(20).view(5, 1, 4)},
                dim=2,
            ),
            {
                "key1": torch.tensor([[[0, 2, 1, 3]]]),
                "key2": torch.tensor(
                    [
                        [[0, 2, 1, 3]],
                        [[4, 6, 5, 7]],
                        [[8, 10, 9, 11]],
                        [[12, 14, 13, 15]],
                        [[16, 18, 17, 19]],
                    ]
                ),
            },
        )


@mark.parametrize("dim", (0, 1, 2, 3))
def test_shuffle_tensor_mapping_dim(dim: int) -> None:
    out = shuffle_tensor_mapping(
        {"key1": torch.rand(1, 2, 3, 4), "key2": torch.ones(1, 2, 3, 4)}, dim=dim
    )
    assert len(out) == 2
    assert out["key1"].shape == (1, 2, 3, 4)
    assert out["key2"].equal(torch.ones(1, 2, 3, 4))


def test_shuffle_tensor_mapping_dim_dict_same_dimension() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping(
                {"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)},
                dim={"key1": 0, "key2": 0},
            ),
            {
                "key1": torch.tensor([0, 2, 1, 3]),
                "key2": torch.tensor(
                    [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]
                ),
            },
        )


def test_shuffle_tensor_mapping_dim_dict_different_dimensions() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping(
                {"key1": torch.arange(4), "key2": torch.arange(20).view(5, 4)},
                dim={"key1": 0, "key2": 1},
            ),
            {
                "key1": torch.tensor([0, 2, 1, 3]),
                "key2": torch.tensor(
                    [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15], [16, 18, 17, 19]]
                ),
            },
        )


def test_shuffle_tensor_mapping_dim_dict_single_tensor() -> None:
    randperm_mock = Mock(return_value=torch.tensor([0, 2, 1, 3]))
    with patch("gravitorch.datapipes.iter.shuffling.torch.randperm", randperm_mock):
        assert objects_are_equal(
            shuffle_tensor_mapping(
                {"key1": torch.arange(4), "key2": torch.arange(20).view(5, 4)}, dim={"key2": 1}
            ),
            {
                "key1": torch.tensor([0, 1, 2, 3]),
                "key2": torch.tensor(
                    [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15], [16, 18, 17, 19]]
                ),
            },
        )


def test_shuffle_tensor_mapping_dim_dict_empty() -> None:
    assert objects_are_equal(
        shuffle_tensor_mapping(
            {"key1": torch.arange(4), "key2": torch.arange(20).view(5, 4)}, dim={}
        ),
        {
            "key1": torch.tensor([0, 1, 2, 3]),
            "key2": torch.tensor(
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
            ),
        },
    )


def test_shuffle_tensor_mapping_incorrect_shape() -> None:
    with raises(ValueError):
        shuffle_tensor_mapping(mapping={"key1": torch.rand(1), "key2": torch.ones(4)})


def test_shuffle_tensor_mapping_same_random_seed() -> None:
    mapping = {"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)}
    assert objects_are_equal(
        shuffle_tensor_mapping(mapping, generator=get_torch_generator(1)),
        shuffle_tensor_mapping(mapping, generator=get_torch_generator(1)),
    )


def test_shuffle_tensor_mapping_different_random_seeds() -> None:
    mapping = {"key1": torch.arange(4), "key2": torch.arange(20).view(4, 5)}
    assert not objects_are_equal(
        shuffle_tensor_mapping(mapping, generator=get_torch_generator(1)),
        shuffle_tensor_mapping(mapping, generator=get_torch_generator(2)),
    )


#########################################
#     Tests for get_first_dimension     #
#########################################


@mark.parametrize("tensor,num_examples", ((torch.ones(4), 4), (torch.ones(5, 4), 5)))
def test_get_first_dimension_torch_tensor(tensor: Tensor, num_examples: int) -> None:
    assert get_first_dimension(tensor) == num_examples


@mark.parametrize("array,num_examples", ((np.ones(4), 4), (np.ones((5, 4)), 5)))
def test_get_first_dimension_numpy_array(array: np.ndarray, num_examples: int) -> None:
    assert get_first_dimension(array) == num_examples


@mark.parametrize(
    "sequence,num_examples",
    (
        ([1, 1, 1], 3),
        ([1, 2, 3, 4], 4),
        ([], 0),
        ((1, 1), 2),
        ((1, 2, 3, 4), 4),
        ((), 0),
    ),
)
def test_get_first_dimension_list_tuple(sequence: Union[list, tuple], num_examples: int) -> None:
    assert get_first_dimension(sequence) == num_examples


@mark.parametrize("obj", (1, "abc", set()))
def test_get_first_dimension_incorrect_type(obj: Any) -> None:
    with raises(TypeError):
        get_first_dimension(obj)
