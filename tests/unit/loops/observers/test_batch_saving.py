from pathlib import Path
from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import BaseEngine
from gravitorch.loops.observers import PyTorchBatchSaver

#######################################
#     Tests for PyTorchBatchSaver     #
#######################################


def test_pytorch_batch_saver_str(tmp_path: Path):
    assert str(PyTorchBatchSaver(tmp_path.joinpath("batch.pt"))).startswith("PyTorchBatchSaver")


def test_pytorch_batch_saver_path(tmp_path: Path):
    path = tmp_path.joinpath("batch.pt")
    assert PyTorchBatchSaver(path)._path == path


@mark.parametrize("max_num_batches", (0, 1, 2))
def test_pytorch_batch_saver_max_num_batches(tmp_path: Path, max_num_batches: int):
    assert (
        PyTorchBatchSaver(tmp_path.joinpath("batch.pt"), max_num_batches)._max_num_batches
        == max_num_batches
    )


def test_pytorch_batch_saver_update_1(tmp_path: Path):
    saver = PyTorchBatchSaver(tmp_path.joinpath("batch.pt"))
    saver.update(engine=Mock(spec=BaseEngine), model_input=1, model_output=2)
    assert saver._batches == [{"model_input": 1, "model_output": 2}]


def test_pytorch_batch_saver_update_2(tmp_path: Path):
    engine = Mock(spec=BaseEngine)
    saver = PyTorchBatchSaver(tmp_path.joinpath("batch.pt"))
    saver.update(engine, model_input=1, model_output=2)
    saver.update(engine, model_input="abc", model_output=123)
    assert saver._batches == [
        {"model_input": 1, "model_output": 2},
        {"model_input": "abc", "model_output": 123},
    ]


def test_pytorch_batch_saver_update_max_num_batches_2(tmp_path: Path):
    engine = Mock(spec=BaseEngine)
    saver = PyTorchBatchSaver(tmp_path.joinpath("batch.pt"), max_num_batches=2)
    saver.update(engine, model_input=1, model_output=11)
    saver.update(engine, model_input=2, model_output=12)
    saver.update(engine, model_input=3, model_output=13)
    assert saver._batches == [
        {"model_input": 1, "model_output": 11},
        {"model_input": 2, "model_output": 12},
    ]


def test_pytorch_batch_saver_start_empty(tmp_path: Path):
    saver = PyTorchBatchSaver(tmp_path.joinpath("batch.pt"))
    saver.start(engine=Mock(spec=BaseEngine))
    assert saver._batches == []


def test_pytorch_batch_saver_reset(tmp_path: Path):
    engine = Mock(spec=BaseEngine)
    saver = PyTorchBatchSaver(tmp_path.joinpath("batch.pt"))
    saver.update(engine, model_input=1, model_output=11)
    saver.update(engine, model_input=2, model_output=12)
    saver.start(engine)
    assert saver._batches == []


def test_pytorch_batch_saver_end(tmp_path: Path):
    path = tmp_path.joinpath("batch.pt")
    engine = Mock(spec=BaseEngine)
    saver = PyTorchBatchSaver(path)
    saver.update(engine, model_input=1, model_output=11)
    saver.update(engine, model_input=2, model_output=12)
    saver.end(engine)
    assert path.is_file()
