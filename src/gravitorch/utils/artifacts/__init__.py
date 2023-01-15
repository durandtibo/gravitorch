__all__ = ["BaseArtifact", "JSONArtifact", "PickleArtifact", "PyTorchArtifact", "TextArtifact"]

from gravitorch.utils.artifacts.base import BaseArtifact
from gravitorch.utils.artifacts.jsona import JSONArtifact
from gravitorch.utils.artifacts.picklea import PickleArtifact
from gravitorch.utils.artifacts.pytorcha import PyTorchArtifact
from gravitorch.utils.artifacts.text import TextArtifact
