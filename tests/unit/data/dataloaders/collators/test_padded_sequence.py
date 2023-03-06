from collections.abc import Hashable
from typing import Union

import torch
from pytest import mark

from gravitorch import constants as ct
from gravitorch.data.dataloaders.collators import (
    DictPaddedSequenceCollator,
    PaddedSequenceCollator,
)

FEATURE = "feature"
INDEX = "index"
MASK = "mask"
NAME = "name"


##################################
#     PaddedSequenceCollator     #
##################################


def test_padded_sequence_collator_collator_str():
    assert str(PaddedSequenceCollator()).startswith("PaddedSequenceCollator(")


@mark.parametrize("length_key", ("length", "abc", 1))
def test_padded_sequence_collator_length_key(length_key: Hashable):
    data = [
        ({length_key: 2}, {FEATURE: torch.full((2,), 2, dtype=torch.float)}),
        ({length_key: 3}, {FEATURE: torch.full((3,), 3, dtype=torch.float)}),
        ({length_key: 4}, {FEATURE: torch.full((4,), 4, dtype=torch.float)}),
    ]
    collator = PaddedSequenceCollator(length_key=length_key)
    batch = collator(data)

    assert len(batch) == 2  # The number of keys in the batch.
    assert torch.equal(batch[length_key], torch.tensor([4, 3, 2]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 3, 2],
            [4, 3, 2],
            [4, 3, 0],
            [4, 0, 0],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


@mark.parametrize("padding_value", [0, 1, -1])
def test_padded_sequence_collator_padding_value(padding_value: float) -> None:
    data = [
        ({ct.LENGTH: 2}, {FEATURE: torch.full((2,), 2, dtype=torch.float)}),
        ({ct.LENGTH: 3}, {FEATURE: torch.full((3,), 3, dtype=torch.float)}),
        ({ct.LENGTH: 4}, {FEATURE: torch.full((4,), 4, dtype=torch.float)}),
    ]
    collator = PaddedSequenceCollator(padding_value=padding_value)
    batch = collator(data)

    assert len(batch) == 2  # The number of keys in the batch.
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 3, 2],
            [4, 3, 2],
            [4, 3, padding_value],
            [4, padding_value, padding_value],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_batch_first_collate_1d():
    data = [
        ({ct.LENGTH: 2}, {FEATURE: torch.full((2,), 2, dtype=torch.float)}),
        ({ct.LENGTH: 3}, {FEATURE: torch.full((3,), 3, dtype=torch.float)}),
        ({ct.LENGTH: 4}, {FEATURE: torch.full((4,), 4, dtype=torch.float)}),
    ]
    collator = PaddedSequenceCollator(batch_first=True)
    batch = collator(data)

    assert len(batch) == 2  # The number of keys in the batch.
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4, 4, 4],
            [3, 3, 3, 0],
            [2, 2, 0, 0],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_2d():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {FEATURE: torch.full((2, 2), 2, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
        ),
    ]
    collator = PaddedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [0, 0]],
            [[4, 4], [0, 0], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_batch_first_collate_2d():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {FEATURE: torch.full((2, 2), 2, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
        ),
    ]
    collator = PaddedSequenceCollator(batch_first=True)
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [4, 4], [4, 4], [4, 4]],
            [[3, 3], [3, 3], [3, 3], [0, 0]],
            [[2, 2], [2, 2], [0, 0], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_3d():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {FEATURE: torch.full((2, 2, 3), 2, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {FEATURE: torch.full((3, 2, 3), 3, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {FEATURE: torch.full((4, 2, 3), 4, dtype=torch.float)},
        ),
    ]
    collator = PaddedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[2, 2, 2], [2, 2, 2]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[2, 2, 2], [2, 2, 2]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[0, 0, 0], [0, 0, 0]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0]]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_batch_first_collate_3d():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {FEATURE: torch.full((2, 2, 3), 2, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {FEATURE: torch.full((3, 2, 3), 3, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {FEATURE: torch.full((4, 2, 3), 4, dtype=torch.float)},
        ),
    ]
    collator = PaddedSequenceCollator(batch_first=True)
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]]],
            [[[3, 3, 3], [3, 3, 3]],
             [[3, 3, 3], [3, 3, 3]],
             [[3, 3, 3], [3, 3, 3]],
             [[0, 0, 0], [0, 0, 0]]],
            [[[2, 2, 2], [2, 2, 2]],
             [[2, 2, 2], [2, 2, 2]],
             [[0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0]]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_empty():
    data = [
        (
            {ct.LENGTH: 0, INDEX: 0, NAME: "item0"},
            {FEATURE: torch.full((0, 2), 2, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
        ),
    ]
    collator = PaddedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [3, 3]],
            [[4, 4], [3, 3]],
            [[4, 4], [3, 3]],
            [[4, 4], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_padded_sequence_collator_mask():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {
                "mask": torch.ones(2, dtype=torch.long),
                FEATURE: torch.full((2, 2), 2, dtype=torch.float),
            },
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {
                "mask": torch.ones(3, dtype=torch.long),
                FEATURE: torch.full((3, 2), 3, dtype=torch.float),
            },
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {
                "mask": torch.ones(4, dtype=torch.long),
                FEATURE: torch.full((4, 2), 4, dtype=torch.float),
            },
        ),
    ]
    collator = PaddedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 5  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [0, 0]],
            [[4, 4], [0, 0], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)
    # fmt: off
    gt_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=torch.long
    )
    # fmt: on
    assert torch.equal(batch["mask"], gt_mask)


######################################
#     DictPaddedSequenceCollator     #
######################################


def test_dict_packed_sequence_collator_str():
    assert str(DictPaddedSequenceCollator(["something"])).startswith("DictPaddedSequenceCollator(")


@mark.parametrize("keys_to_pad", (("key", 1), ["key", 1]))
def test_dict_packed_sequence_collator_keys_to_pad(
    keys_to_pad: Union[list[Hashable], tuple[Hashable, ...]]
) -> None:
    assert DictPaddedSequenceCollator(keys_to_pad)._keys_to_pad == ("key", 1)


def test_dict_padded_sequence_collator_1d():
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE])
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 3, 2],
            [4, 3, 2],
            [4, 3, 0],
            [4, 0, 0],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_batch_first_1d():
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE], batch_first=True)
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4, 4, 4],
            [3, 3, 3, 0],
            [2, 2, 0, 0],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_2d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [2, 2]],
            [[4, 4], [3, 3], [0, 0]],
            [[4, 4], [0, 0], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_batch_first_2d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE], batch_first=True)
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [4, 4], [4, 4], [4, 4]],
            [[3, 3], [3, 3], [3, 3], [0, 0]],
            [[2, 2], [2, 2], [0, 0], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_3d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2, 3), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2, 3), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2, 3), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[2, 2, 2], [2, 2, 2]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[2, 2, 2], [2, 2, 2]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[3, 3, 3], [3, 3, 3]],
             [[0, 0, 0], [0, 0, 0]]],
            [[[4, 4, 4], [4, 4, 4]],
             [[0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0]]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_batch_first_3d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2, 3), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2, 3), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2, 3), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE], batch_first=True)
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]],
             [[4, 4, 4], [4, 4, 4]]],
            [[[3, 3, 3], [3, 3, 3]],
             [[3, 3, 3], [3, 3, 3]],
             [[3, 3, 3], [3, 3, 3]],
             [[0, 0, 0], [0, 0, 0]]],
            [[[2, 2, 2], [2, 2, 2]],
             [[2, 2, 2], [2, 2, 2]],
             [[0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 0]]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


@mark.parametrize("padding_value", [0, 1, -1])
def test_dict_padded_sequence_collator_padding_value(padding_value: float) -> None:
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE], padding_value=padding_value)
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 3, 2],
            [4, 3, 2],
            [4, 3, padding_value],
            [4, padding_value, padding_value],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_empty():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((0, 2), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1"]
    assert torch.equal(batch[INDEX], torch.tensor([2, 1]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4], [3, 3]],
            [[4, 4], [3, 3]],
            [[4, 4], [3, 3]],
            [[4, 4], [0, 0]],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)


def test_dict_padded_sequence_collator_ignore_extra_key():
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPaddedSequenceCollator(keys_to_pad=[FEATURE, INDEX])
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 3, 2],
            [4, 3, 2],
            [4, 3, 0],
            [4, 0, 0],
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE], gt_feature)
