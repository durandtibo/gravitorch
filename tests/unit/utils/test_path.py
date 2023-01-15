import os
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

from pytest import fixture, raises

from gravitorch.utils.io import save_text
from gravitorch.utils.path import (
    find_tar_files,
    get_human_readable_file_size,
    get_number_of_files,
    get_original_cwd,
    get_pythonpath,
    sanitize_path,
    working_directory,
)

######################################
#     Tests for get_original_cwd     #
######################################


def test_get_original_cwd_no_hydra():
    assert get_original_cwd() == Path.cwd()


@patch("hydra.core.hydra_config.HydraConfig.initialized", lambda *args, **kwargs: True)
def test_get_original_cwd_with_hydra(tmp_path: Path):
    with patch("hydra.utils.get_original_cwd", lambda *args, **kwargs: tmp_path.as_posix()):
        assert get_original_cwd() == tmp_path


####################################
#     Tests for get_pythonpath     #
####################################


def test_get_pythonpath_defined(tmp_path: Path):
    with patch.dict(os.environ, {"PYTHONPATH": tmp_path.as_posix()}, clear=True):
        assert get_pythonpath() == tmp_path


@patch.dict(os.environ, {}, clear=True)
def test_get_pythonpath_not_defined(tmp_path: Path):
    with patch("gravitorch.utils.path.get_original_cwd", lambda *args, **kwargs: tmp_path):
        assert get_pythonpath() == tmp_path


#######################################
#     Tests for working_directory     #
#######################################


def test_working_directory():
    cwd_before = Path.cwd()
    new_path = cwd_before.parent
    with working_directory(new_path):
        assert Path.cwd() == new_path

    assert Path.cwd() == cwd_before


def test_working_directory_error():
    cwd_before = Path.cwd()
    with raises(RuntimeError):
        with working_directory(cwd_before.parent):
            raise RuntimeError

    assert Path.cwd() == cwd_before


#########################################
#     Tests for get_number_of_files     #
#########################################


def test_get_number_of_files(tmp_path: Path):
    save_text("", tmp_path.joinpath("file1.txt"))
    save_text("", tmp_path.joinpath("file2.txt"))
    save_text("", tmp_path.joinpath("subdir", "file.txt"))
    save_text("", tmp_path.joinpath("subdir", "subdir", "file.txt"))
    assert get_number_of_files(tmp_path.as_posix()) == 4


def test_get_number_of_files_empty(tmp_path: Path):
    assert get_number_of_files(tmp_path.as_posix()) == 0


####################################
#     Tests for find_tar_files     #
####################################


@fixture
def tar_path(tmp_path: Path) -> Path:
    save_text("text", tmp_path.joinpath("a.txt"))
    tmp_path.joinpath("subfolder", "sub").mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_path.joinpath("data.tar"), "w") as tar:
        tar.add(tmp_path.joinpath("a.txt"))
    with tarfile.open(tmp_path.joinpath("data2.tar.gz"), "w:gz") as tar:
        tar.add(tmp_path.joinpath("a.txt"))
    with tarfile.open(tmp_path.joinpath("subfolder", "data.tar"), "w") as tar:
        tar.add(tmp_path.joinpath("a.txt"))
    with tarfile.open(tmp_path.joinpath("subfolder", "sub", "data.tar"), "w") as tar:
        tar.add(tmp_path.joinpath("a.txt"))
    return tmp_path


def test_find_tar_files(tar_path: Path):
    assert sorted(find_tar_files(tar_path)) == [
        tar_path.joinpath("data.tar"),
        tar_path.joinpath("data2.tar.gz"),
        tar_path.joinpath("subfolder", "data.tar"),
        tar_path.joinpath("subfolder", "sub", "data.tar"),
    ]


def test_find_tar_files_recursive_false(tar_path: Path):
    paths = sorted(find_tar_files(tar_path, recursive=False))
    assert len(paths) == 2
    assert paths == [tar_path.joinpath("data.tar"), tar_path.joinpath("data2.tar.gz")]


def test_find_tar_files_file(tar_path: Path):
    paths = sorted(find_tar_files(tar_path.joinpath("subfolder", "sub", "data.tar")))
    assert paths == [tar_path.joinpath("subfolder", "sub", "data.tar")]


def test_find_tar_files_empty(tmp_path: Path):
    save_text("text", tmp_path.joinpath("file.txt"))
    assert find_tar_files(tmp_path.joinpath("file.txt")) == tuple()


###################################
#     Tests for sanitize_path     #
###################################


def test_sanitize_path_empty_str():
    assert sanitize_path("") == Path.cwd()


def test_sanitize_path_str():
    assert sanitize_path("something") == Path.cwd().joinpath("something")


def test_sanitize_path_path(tmp_path: Path):
    assert sanitize_path(tmp_path) == tmp_path


def test_sanitize_path_resolve():
    assert sanitize_path(Path("something/./../")) == Path.cwd()


def test_sanitize_path_uri():
    assert sanitize_path("file:///my/path/something/./../") == Path("/my/path")


##################################################
#     Tests for get_human_readable_file_size     #
##################################################


def test_get_human_readable_file_size(tmp_path: Path):
    path = tmp_path.joinpath("data.txt")
    save_text("", path)
    assert get_human_readable_file_size(path, unit="MB") == "0.00 MB"


def test_get_human_readable_file_size_2kb():
    path = Mock(spec=Path, stat=Mock(return_value=Mock(st_size=2048)))
    sanitize_mock = Mock(return_value=path)
    with patch("gravitorch.utils.path.sanitize_path", sanitize_mock):
        assert get_human_readable_file_size(path, unit="KB") == "2.00 KB"
        sanitize_mock.assert_called_once_with(path)
