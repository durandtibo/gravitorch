from pathlib import Path
from unittest.mock import Mock, patch

from pytest import mark, raises

from gravitorch.data.datapipes.iter import PickleSaver, PyTorchSaver, SourceWrapper
from gravitorch.utils.io import save_text

#################################
#     Tests for PickleSaver     #
#################################


def test_pickle_saver_str(tmp_path: Path) -> None:
    assert str(PickleSaver(SourceWrapper([]), root_path=tmp_path)).startswith(
        "PickleSaverIterDataPipe("
    )


def test_pickle_saver_incorrect_root_path(tmp_path: Path) -> None:
    root_path = tmp_path.joinpath("file.txt")
    save_text("abc", root_path)
    with raises(ValueError):
        PickleSaver(SourceWrapper([1, 2]), root_path=root_path)


def test_pickle_saver_incorrect_pattern(tmp_path: Path) -> None:
    with raises(ValueError):
        PickleSaver(SourceWrapper([1, 2]), root_path=tmp_path, pattern="data.pkl")


@mark.parametrize("num_samples", (1, 2))
def test_pickle_saver_iter(tmp_path: Path, num_samples: int) -> None:
    datapipe = PickleSaver(SourceWrapper(list(range(num_samples))), root_path=tmp_path)
    files = list(datapipe)
    assert files == [tmp_path.joinpath(f"data_{i:04d}.pkl") for i in range(num_samples)]
    for file in files:
        assert file.is_file()


def test_pickle_saver_iter_pattern(tmp_path: Path) -> None:
    datapipe = PickleSaver(SourceWrapper([1, 2]), root_path=tmp_path, pattern="file{index}.pkl")
    files = list(datapipe)
    assert files == [tmp_path.joinpath("file0.pkl"), tmp_path.joinpath("file1.pkl")]
    for file in files:
        assert file.is_file()


def test_pickle_saver_iter_file(tmp_path: Path) -> None:
    datapipe = PickleSaver(SourceWrapper([1]), root_path=tmp_path)
    with patch("gravitorch.data.datapipes.iter.saving.save_pickle") as save_mock:
        list(datapipe)
        save_mock.assert_called_once_with(1, tmp_path.joinpath("data_0000.pkl"))


def test_pickle_saver_len(tmp_path: Path) -> None:
    assert len(PickleSaver(Mock(__len__=Mock(return_value=5)), root_path=tmp_path)) == 5


def test_pickle_saver_no_len(tmp_path: Path) -> None:
    with raises(TypeError):
        len(PickleSaver(SourceWrapper(Mock()), root_path=tmp_path))


##################################
#     Tests for PyTorchSaver     #
##################################


def test_pytorch_saver_str(tmp_path: Path) -> None:
    assert str(PyTorchSaver(SourceWrapper([]), root_path=tmp_path)).startswith(
        "PyTorchSaverIterDataPipe("
    )


def test_pytorch_saver_incorrect_root_path(tmp_path: Path) -> None:
    root_path = tmp_path.joinpath("file.txt")
    save_text("abc", root_path)
    with raises(ValueError):
        PyTorchSaver(SourceWrapper([1, 2]), root_path=root_path)


def test_pytorch_saver_incorrect_pattern(tmp_path: Path) -> None:
    with raises(ValueError):
        PyTorchSaver(SourceWrapper([1, 2]), root_path=tmp_path, pattern="data.pt")


@mark.parametrize("num_samples", (1, 2))
def test_pytorch_saver_iter(tmp_path: Path, num_samples: int) -> None:
    datapipe = PyTorchSaver(SourceWrapper(list(range(num_samples))), root_path=tmp_path)
    files = list(datapipe)
    assert files == [tmp_path.joinpath(f"data_{i:04d}.pt") for i in range(num_samples)]
    for file in files:
        assert file.is_file()


def test_pytorch_saver_iter_pattern(tmp_path: Path) -> None:
    datapipe = PyTorchSaver(SourceWrapper([1, 2]), root_path=tmp_path, pattern="file{index}.pt")
    files = list(datapipe)
    assert files == [tmp_path.joinpath("file0.pt"), tmp_path.joinpath("file1.pt")]
    for file in files:
        assert file.is_file()


def test_pytorch_saver_iter_file(tmp_path: Path) -> None:
    datapipe = PyTorchSaver(SourceWrapper([1]), root_path=tmp_path)
    with patch("gravitorch.data.datapipes.iter.saving.save_pytorch") as save_mock:
        list(datapipe)
        save_mock.assert_called_once_with(1, tmp_path.joinpath("data_0000.pt"))


def test_pytorch_saver_len(tmp_path: Path) -> None:
    assert len(PyTorchSaver(Mock(__len__=Mock(return_value=5)), root_path=tmp_path)) == 5


def test_pytorch_saver_no_len(tmp_path: Path) -> None:
    with raises(TypeError):
        len(PyTorchSaver(SourceWrapper(Mock()), root_path=tmp_path))
