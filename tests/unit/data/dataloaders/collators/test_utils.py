from objectory import OBJECT_TARGET
from torch.utils.data.dataloader import default_collate

from gravitorch.data.dataloaders.collators import PaddedSequenceCollator, setup_collator

####################################
#     Tests for setup_collator     #
####################################


def test_setup_collator_object():
    collator = PaddedSequenceCollator()
    assert setup_collator(collator) is collator


def test_setup_collator_dict():
    assert isinstance(
        setup_collator(
            {OBJECT_TARGET: "gravitorch.data.dataloaders.collators.PaddedSequenceCollator"}
        ),
        PaddedSequenceCollator,
    )


def test_setup_collator_none():
    assert setup_collator(None) == default_collate
