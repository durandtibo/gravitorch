r"""This package contains the implementation of some data loader
collators."""

from __future__ import annotations

__all__ = [
    "BaseCollator",
    "DictPackedSequenceCollator",
    "DictPaddedSequenceCollator",
    "PackedSequenceCollator",
    "PaddedSequenceCollator",
    "setup_collator",
]

from gravitorch.dataloaders.collators.base import BaseCollator, setup_collator
from gravitorch.dataloaders.collators.pack import (
    DictPackedSequenceCollator,
    PackedSequenceCollator,
)
from gravitorch.dataloaders.collators.pad import (
    DictPaddedSequenceCollator,
    PaddedSequenceCollator,
)
