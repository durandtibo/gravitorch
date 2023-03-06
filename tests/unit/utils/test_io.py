import json
import pickle
from pathlib import Path

import numpy as np
import torch
from coola import objects_are_equal
from pytest import fixture, mark, raises

from gravitorch.utils.io import (
    load_json,
    load_pickle,
    load_text,
    load_yaml,
    save_json,
    save_pickle,
    save_pytorch,
    save_text,
    save_yaml,
)


@fixture
def to_save() -> dict:
    return {
        "string": "oui oui",
        123: "abc",
        "int": 1,
        "float": 2.33,
        "list": [1, 2, 3],
        "dict": {"abc": 123},
        # torch tensor
        "torch_tensor_0": torch.tensor(0),
        "torch_tensor_0.5": torch.tensor(0.5),
        "torch_tensor_1": torch.tensor([1, 2, 3]),
        "torch_tensor_2": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        # numpy
        "np_int32": np.int32(1),
        "np_int64": np.int64(2),
        "np_float32": np.float32(1.5),
        "np_float64": np.float64(2.5),
        "np_array_0": np.array(0),
        "np_array_0.5": np.array(0.5),
        "np_array_1": np.array([1, 2, 3]),
        "np_array_2": np.array([[1, 2, 3], [4, 5, 6]]),
        "np_datetime64": np.datetime64("2020-01-01"),
    }


@fixture
def json_data() -> dict:
    return {
        "string": "oui oui",
        "123": "abc",
        "int": 1,
        "float": 2.33,
        "list": [1, 2, 3],
        "dict": {"abc": 123},
        # torch tensor
        "torch_tensor_0": 0,
        "torch_tensor_0.5": 0.5,
        "torch_tensor_1": [1, 2, 3],
        "torch_tensor_2": [[1, 2, 3], [4, 5, 6]],
        # numpy
        "np_int32": 1,
        "np_int64": 2,
        "np_float32": 1.5,
        "np_float64": 2.5,
        "np_array_0": 0,
        "np_array_0.5": 0.5,
        "np_array_1": [1, 2, 3],
        "np_array_2": [[1, 2, 3], [4, 5, 6]],
        "np_datetime64": "2020-01-01",
    }


@fixture
def yaml_data() -> dict:
    return {
        "string": "oui oui",
        123: "abc",
        "int": 1,
        "float": 2.33,
        "list": [1, 2, 3],
        "dict": {"abc": 123},
        # torch tensor
        "torch_tensor_0": 0,
        "torch_tensor_0.5": 0.5,
        "torch_tensor_1": [1, 2, 3],
        "torch_tensor_2": [[1, 2, 3], [4, 5, 6]],
        # numpy
        "np_int32": 1,
        "np_int64": 2,
        "np_float32": 1.5,
        "np_float64": 2.5,
        "np_array_0": 0,
        "np_array_0.5": 0.5,
        "np_array_1": [1, 2, 3],
        "np_array_2": [[1, 2, 3], [4, 5, 6]],
        "np_datetime64": "2020-01-01",
    }


def test_save_load_text(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data", "data.txt")
    save_text("Hello!", file_path)
    assert load_text(file_path) == "Hello!"


################
#     JSON     #
################


def test_load_json(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data.json")
    to_save = {"abc": 123, 123: "abc"}
    with Path.open(file_path, "w") as file:
        json.dump(to_save, file)

    assert load_json(file_path) == {"abc": 123, "123": "abc"}


def test_load_json_missing(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data.json")
    with raises(FileNotFoundError):
        load_json(file_path)


def test_save_json(tmp_path: Path, to_save: dict, json_data: dict) -> None:
    file_path = tmp_path.joinpath("data", "data.json")
    save_json(to_save, file_path)

    assert file_path.is_file()
    with Path.open(file_path, mode="rb") as file:
        data = json.load(file)
    assert data == json_data


##################
#     Pickle     #
##################


def test_load_pickle(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data.pkl")
    to_save = {"abc": 123, 123: "abc"}
    with Path.open(file_path, "wb") as file:
        pickle.dump(to_save, file)

    data = load_pickle(file_path)
    assert data == {"abc": 123, 123: "abc"}


def test_load_pickle_missing(tmp_path: Path) -> None:
    file_path = tmp_path.joinpath("data.pkl")
    with raises(FileNotFoundError):
        load_pickle(file_path)


@mark.parametrize("protocol", [1, 2, 3, 4])
def test_save_pickle(tmp_path: Path, protocol: int, to_save: dict) -> None:
    file_path = tmp_path.joinpath("data", "data.pkl")
    save_pickle(to_save, file_path, protocol)
    assert file_path.is_file()
    with Path.open(file_path, mode="rb") as file:
        data = pickle.load(file)
    assert objects_are_equal(data, to_save)


###################
#     PyTorch     #
###################


def test_save_pytorch(tmp_path: Path, to_save: dict) -> None:
    file_path = tmp_path.joinpath("data", "data.pt")
    save_pytorch(to_save, file_path)
    assert file_path.is_file()
    assert objects_are_equal(torch.load(file_path), to_save)


################
#     YAML     #
################


def test_save_load_yaml(tmp_path: Path, to_save: dict, yaml_data: dict) -> None:
    file_path = tmp_path.joinpath("data", "data.yaml")
    save_yaml(to_save, file_path)
    assert load_yaml(file_path) == yaml_data
