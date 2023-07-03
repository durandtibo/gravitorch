from __future__ import annotations

__all__ = [
    "clone_datapipe",
    "setup_datapipe",
    "is_iter_datapipe_config",
    "setup_iter_datapipe",
]

from gravitorch.datapipes.factory import setup_datapipe
from gravitorch.datapipes.iter import is_iter_datapipe_config, setup_iter_datapipe
from gravitorch.datapipes.utils import clone_datapipe
