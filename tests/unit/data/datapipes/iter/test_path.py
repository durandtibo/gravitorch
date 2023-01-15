from pathlib import Path
from unittest.mock import Mock

from pytest import raises
from torch.utils.data import IterDataPipe

from gravitorch.data.datapipes.iter import (
    DirFilterIterDataPipe,
    FileFilterIterDataPipe,
    PathListerIterDataPipe,
    SourceIterDataPipe,
)
from gravitorch.utils.io import save_text


def create_files(path: Path) -> None:
    save_text("", path.joinpath("file.txt"))
    save_text("", path.joinpath("dir/file.txt"))


############################################
#     Tests for DirFilterIterDataPipe     #
############################################


def test_dir_filter_iter_datapipe_str():
    assert str(DirFilterIterDataPipe(Mock(spec=IterDataPipe))).startswith("DirFilterIterDataPipe(")


def test_dir_filter_iter_datapipe_iter_recursive(tmp_path: Path):
    create_files(tmp_path)
    assert list(
        DirFilterIterDataPipe(
            SourceIterDataPipe(
                [
                    tmp_path.joinpath("dir/file.txt"),
                    tmp_path.joinpath("dir/"),
                    tmp_path.joinpath("file.txt"),
                ]
            )
        )
    ) == [tmp_path.joinpath("dir/")]


def test_dir_filter_iter_datapipe_len(tmp_path: Path):
    with raises(TypeError):
        len(DirFilterIterDataPipe(Mock(spec=IterDataPipe)))


############################################
#     Tests for FileFilterIterDataPipe     #
############################################


def test_file_filter_iter_datapipe_str():
    assert str(FileFilterIterDataPipe(Mock(spec=IterDataPipe))).startswith(
        "FileFilterIterDataPipe("
    )


def test_file_filter_iter_datapipe_iter_recursive(tmp_path: Path):
    create_files(tmp_path)
    assert list(
        FileFilterIterDataPipe(
            SourceIterDataPipe(
                [
                    tmp_path.joinpath("dir/file.txt"),
                    tmp_path.joinpath("dir/"),
                    tmp_path.joinpath("file.txt"),
                ]
            )
        )
    ) == [
        tmp_path.joinpath("dir/file.txt"),
        tmp_path.joinpath("file.txt"),
    ]


def test_file_filter_iter_datapipe_len(tmp_path: Path):
    with raises(TypeError):
        len(FileFilterIterDataPipe(Mock(spec=IterDataPipe)))


############################################
#     Tests for PathListerIterDataPipe     #
############################################


def test_path_lister_iter_datapipe_str():
    assert str(PathListerIterDataPipe(Mock(spec=IterDataPipe))).startswith(
        "PathListerIterDataPipe("
    )


def test_path_lister_iter_datapipe_iter_empty(tmp_path: Path):
    assert list(PathListerIterDataPipe(SourceIterDataPipe([tmp_path]))) == []


def test_path_lister_iter_datapipe_iter(tmp_path: Path):
    create_files(tmp_path)
    assert list(PathListerIterDataPipe(SourceIterDataPipe([tmp_path]), pattern="*.txt")) == [
        tmp_path.joinpath("file.txt")
    ]


def test_path_lister_iter_datapipe_iter_recursive(tmp_path: Path):
    create_files(tmp_path)
    assert list(PathListerIterDataPipe(SourceIterDataPipe([tmp_path]), pattern="**/*.txt")) == [
        tmp_path.joinpath("dir/file.txt"),
        tmp_path.joinpath("file.txt"),
    ]


def test_path_lister_iter_datapipe_iter_deterministic_false(tmp_path: Path):
    create_files(tmp_path)
    assert set(
        PathListerIterDataPipe(
            SourceIterDataPipe([tmp_path]), pattern="**/*.txt", deterministic=False
        )
    ) == {
        tmp_path.joinpath("dir/file.txt"),
        tmp_path.joinpath("file.txt"),
    }


def test_path_lister_iter_datapipe_len(tmp_path: Path):
    with raises(TypeError):
        len(PathListerIterDataPipe(tmp_path))
