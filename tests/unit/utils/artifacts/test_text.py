from pathlib import Path

from gravitorch.utils.artifacts import TextArtifact
from gravitorch.utils.io import load_text

##################################
#     Tests for TextArtifact     #
##################################


def test_text_artifact_str() -> None:
    assert str(TextArtifact(tag="name", data="abc")).startswith("TextArtifact(")


def test_text_artifact_create(tmp_path: Path) -> None:
    TextArtifact(tag="name", data="abc").create(tmp_path)
    path = tmp_path.joinpath("name.txt")
    assert path.is_file()
    assert load_text(path) == "abc"
