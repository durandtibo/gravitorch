r"""This package contains the implementation of some data loader collators."""

__all__ = [
    "BaseCollator",
    "DictPackedSequenceCollator",
    "DictPaddedSequenceCollator",
    "PackedSequenceCollator",
    "PaddedSequenceCollator",
    "setup_collator",
]

from gravitorch.data.dataloaders.collators.base import BaseCollator
from gravitorch.data.dataloaders.collators.packed_sequence import (
    DictPackedSequenceCollator,
    PackedSequenceCollator,
)
from gravitorch.data.dataloaders.collators.padded_sequence import (
    DictPaddedSequenceCollator,
    PaddedSequenceCollator,
)
from gravitorch.data.dataloaders.collators.utils import setup_collator
