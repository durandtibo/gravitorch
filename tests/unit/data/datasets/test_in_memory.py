from collections.abc import Sequence
from pathlib import Path

from pytest import mark

from gravitorch.data.datasets import ExampleDataset
from gravitorch.utils.io import save_json, save_pickle, save_pytorch

#####################################
#     Tests for ExampleDataset     #
#####################################


def test_example_dataset_str() -> None:
    assert str(ExampleDataset(())).startswith("ExampleDataset")


def test_example_dataset_examples_list_to_tuple() -> None:
    assert ExampleDataset([1, 2])._examples == (1, 2)


@mark.parametrize("examples,length", (((), 0), ((1,), 1), ((1, 2), 2)))
def test_example_dataset_len(examples: Sequence, length: int) -> None:
    assert len(ExampleDataset(examples)) == length


def test_example_dataset_getitem() -> None:
    dataset = ExampleDataset((1, 2))
    assert dataset[0] == 1
    assert dataset[1] == 2


def test_example_dataset_equal_true() -> None:
    assert ExampleDataset([1, 2]).equal(ExampleDataset([1, 2]))


def test_example_dataset_equal_true_same() -> None:
    dataset = ExampleDataset([1, 2])
    assert dataset.equal(dataset)


def test_example_dataset_equal_false_different_examples() -> None:
    assert not ExampleDataset([1, 2]).equal(ExampleDataset([2, 1]))


def test_example_dataset_equal_false_different_type() -> None:
    assert not ExampleDataset([1, 2]).equal([1, 2])


def test_example_dataset_from_json_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_json((1, 2), path)
    assert ExampleDataset.from_json_file(path).equal(ExampleDataset((1, 2)))


def test_example_dataset_from_pickle_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle((1, 2), path)
    assert ExampleDataset.from_pickle_file(path).equal(ExampleDataset((1, 2)))


def test_example_dataset_from_pytorch_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch((1, 2), path)
    assert ExampleDataset.from_pytorch_file(path).equal(ExampleDataset((1, 2)))
