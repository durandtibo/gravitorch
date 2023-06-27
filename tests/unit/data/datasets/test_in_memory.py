from collections.abc import Sequence
from pathlib import Path

from pytest import mark, raises

from gravitorch.data.datasets import InMemoryDataset
from gravitorch.data.datasets.in_memory import FileToInMemoryDataset, _load_examples
from gravitorch.utils.io import save_json, save_pickle, save_pytorch

#####################################
#     Tests for InMemoryDataset     #
#####################################


def test_in_memory_dataset_str() -> None:
    assert str(InMemoryDataset(())).startswith("InMemoryDataset")


def test_in_memory_dataset_examples_list_to_tuple() -> None:
    assert InMemoryDataset([1, 2])._examples == (1, 2)


@mark.parametrize("examples,length", (((), 0), ((1,), 1), ((1, 2), 2)))
def test_in_memory_dataset_len(examples: Sequence, length: int) -> None:
    assert len(InMemoryDataset(examples)) == length


def test_in_memory_dataset_getitem() -> None:
    dataset = InMemoryDataset((1, 2))
    assert dataset[0] == 1
    assert dataset[1] == 2


def test_in_memory_dataset_equal_true() -> None:
    assert InMemoryDataset([1, 2]).equal(InMemoryDataset([1, 2]))


def test_in_memory_dataset_equal_true_same() -> None:
    dataset = InMemoryDataset([1, 2])
    assert dataset.equal(dataset)


def test_in_memory_dataset_equal_false_different_examples() -> None:
    assert not InMemoryDataset([1, 2]).equal(InMemoryDataset([2, 1]))


def test_in_memory_dataset_equal_false_different_type() -> None:
    assert not InMemoryDataset([1, 2]).equal([1, 2])


def test_in_memory_dataset_from_json_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_json((1, 2), path)
    assert InMemoryDataset.from_json_file(path).equal(InMemoryDataset((1, 2)))


def test_in_memory_dataset_from_pickle_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle((1, 2), path)
    assert InMemoryDataset.from_pickle_file(path).equal(InMemoryDataset((1, 2)))


def test_in_memory_dataset_from_pytorch_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch((1, 2), path)
    assert InMemoryDataset.from_pytorch_file(path).equal(InMemoryDataset((1, 2)))


###########################################
#     Tests for FileToInMemoryDataset     #
###########################################


def test_file_to_in_memory_dataset_str(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch((1, 2), path)
    assert str(FileToInMemoryDataset(path)).startswith("FileToInMemoryDataset")


@mark.parametrize("examples,length", (((), 0), ((1,), 1), ((1, 2), 2)))
def test_file_to_in_memory_dataset_len(tmp_path: Path, examples: Sequence, length: int) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch(examples, path)
    assert len(FileToInMemoryDataset(path)) == length


def test_file_to_in_memory_dataset_getitem(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch((1, 2), path)
    dataset = FileToInMemoryDataset(path)
    assert dataset[0] == 1
    assert dataset[1] == 2


####################################
#     Tests for _load_examples     #
####################################


@mark.parametrize("examples", ((1, 2, 3), [1, 2, 3]))
def test_load_examples_json(tmp_path: Path, examples: Sequence) -> None:
    path = tmp_path.joinpath("data.json")
    save_json(examples, path)
    assert _load_examples(path) == (1, 2, 3)


@mark.parametrize("examples", ((1, 2, 3), [1, 2, 3]))
def test_load_examples_pkl(tmp_path: Path, examples: Sequence) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle(examples, path)
    assert _load_examples(path) == (1, 2, 3)


@mark.parametrize("examples", ((1, 2, 3), [1, 2, 3]))
def test_load_examples_pt(tmp_path: Path, examples: Sequence) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch(examples, path)
    assert _load_examples(path) == (1, 2, 3)


def test_load_examples_incorrect_extension(tmp_path: Path) -> None:
    with raises(ValueError, match="Incorrect file extension '.something'"):
        _load_examples(tmp_path.joinpath("data.something"))
