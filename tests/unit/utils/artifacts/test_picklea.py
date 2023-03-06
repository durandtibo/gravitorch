from pathlib import Path
from unittest.mock import patch

from pytest import mark

from gravitorch.utils.artifacts import PickleArtifact
from gravitorch.utils.io import load_pickle

####################################
#     Tests for PickleArtifact     #
####################################


def test_pickle_artifact_str() -> None:
    assert str(PickleArtifact(tag="name", data={"a": 1, "b": 2})).startswith("PickleArtifact(")


def test_pickle_artifact_create(tmp_path: Path) -> None:
    PickleArtifact(tag="name", data={"a": 1, "b": 2}).create(tmp_path)
    path = tmp_path.joinpath("name.pkl")
    assert path.is_file()
    assert load_pickle(path) == {"a": 1, "b": 2}


@mark.parametrize("protocol", (4, 5))
def test_pickle_artifact_create_protocol(tmp_path: Path, protocol: int) -> None:
    with patch("gravitorch.utils.artifacts.picklea.save_pickle") as save_mock:
        PickleArtifact(tag="name", data={"a": 1, "b": 2}, protocol=protocol).create(tmp_path)
        save_mock.assert_called_once_with(
            to_save={"a": 1, "b": 2},
            path=tmp_path.joinpath("name.pkl"),
            protocol=protocol,
        )
