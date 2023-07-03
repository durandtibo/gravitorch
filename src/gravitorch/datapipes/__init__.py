from __future__ import annotations

__all__ = [
    "clone_datapipe",
    "setup_datapipe",
    "is_iterdatapipe_config",
    "setup_iterdatapipe",
]

from gravitorch.datapipes.factory import setup_datapipe
from gravitorch.datapipes.iter import is_iterdatapipe_config, setup_iterdatapipe
from gravitorch.datapipes.utils import clone_datapipe
