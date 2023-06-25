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

from gravitorch.data.dataloaders.collators.base import BaseCollator
from gravitorch.data.dataloaders.collators.pack import (
    DictPackedSequenceCollator,
    PackedSequenceCollator,
)
from gravitorch.data.dataloaders.collators.pad import (
    DictPaddedSequenceCollator,
    PaddedSequenceCollator,
)
from gravitorch.data.dataloaders.collators.utils import setup_collator
