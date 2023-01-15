from pathlib import Path

import torch

from gravitorch.utils.artifacts import PyTorchArtifact

#####################################
#     Tests for PyTorchArtifact     #
#####################################


def test_pytorch_artifact_str():
    assert str(PyTorchArtifact(tag="name", data={"a": 1, "b": 2})).startswith("PyTorchArtifact(")


def test_pytorch_artifact_create(tmp_path: Path):
    PyTorchArtifact(tag="name", data={"a": 1, "b": 2}).create(tmp_path)
    path = tmp_path.joinpath("name.pt")
    assert path.is_file()
    assert torch.load(path) == {"a": 1, "b": 2}
