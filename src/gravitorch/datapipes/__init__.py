from __future__ import annotations

__all__ = [
    "clone_datapipe",
    "create_chained_datapipe",
    "is_datapipe_config",
    "is_iter_datapipe_config",
    "is_map_datapipe_config",
    "setup_datapipe",
    "setup_iter_datapipe",
    "setup_map_datapipe",
]

from gravitorch.datapipes.factory import (
    create_chained_datapipe,
    is_datapipe_config,
    setup_datapipe,
)
from gravitorch.datapipes.iter import is_iter_datapipe_config, setup_iter_datapipe
from gravitorch.datapipes.map import is_map_datapipe_config, setup_map_datapipe
from gravitorch.datapipes.utils import clone_datapipe
