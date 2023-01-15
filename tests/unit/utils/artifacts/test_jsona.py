from pathlib import Path

from gravitorch.utils.artifacts import JSONArtifact
from gravitorch.utils.io import load_json

##################################
#     Tests for JSONArtifact     #
##################################


def test_json_artifact_str():
    assert str(JSONArtifact(tag="name", data={"a": 1, "b": 2})).startswith("JSONArtifact(")


def test_json_artifact_create(tmp_path: Path):
    JSONArtifact(tag="name", data={"a": 1, "b": 2}).create(tmp_path)
    path = tmp_path.joinpath("name.json")
    assert path.is_file()
    assert load_json(path) == {"a": 1, "b": 2}
