r"""The data package contains the data loader base class and some tools
to speed up the implementation or setup of new data loaders."""

from __future__ import annotations

__all__ = [
    "create_dataloader",
    "create_dataloader2",
    "is_dataloader_config",
    "is_dataloader2_config",
    "setup_dataloader",
]

from gravitorch.data.dataloaders.factory import (
    create_dataloader,
    create_dataloader2,
    is_dataloader2_config,
    is_dataloader_config,
    setup_dataloader,
)
