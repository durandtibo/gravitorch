from __future__ import annotations

from objectory import OBJECT_TARGET
from torch.utils.data.dataloader import default_collate

from gravitorch.data.dataloaders.collators import PaddedSequenceCollator, setup_collator

####################################
#     Tests for setup_collator     #
####################################


def test_setup_collator_object() -> None:
    collator = PaddedSequenceCollator()
    assert setup_collator(collator) is collator


def test_setup_collator_dict() -> None:
    assert isinstance(
        setup_collator(
            {OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
        ),
        PaddedSequenceCollator,
    )


def test_setup_collator_none() -> None:
    assert setup_collator(None) == default_collate
