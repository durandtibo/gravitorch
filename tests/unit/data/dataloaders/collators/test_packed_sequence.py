from collections.abc import Hashable
from typing import Union

import torch
from pytest import mark

from gravitorch import constants as ct
from gravitorch.data.dataloaders.collators import (
    DictPackedSequenceCollator,
    PackedSequenceCollator,
)

FEATURE = "feature"
INDEX = "index"
MASK = "mask"
NAME = "name"


##################################
#     PackedSequenceCollator     #
##################################


def test_packed_sequence_collator_str():
    assert str(PackedSequenceCollator()).startswith("PackedSequenceCollator(")


@mark.parametrize("length_key", ("length", "abc", 1))
def test_packed_sequence_collator_length_key(length_key: Hashable):
    data = [
        ({length_key: 2}, {FEATURE: torch.full((2,), 2, dtype=torch.float)}),
        ({length_key: 3}, {FEATURE: torch.full((3,), 3, dtype=torch.float)}),
        ({length_key: 4}, {FEATURE: torch.full((4,), 4, dtype=torch.float)}),
    ]
    collator = PackedSequenceCollator(length_key=length_key)
    batch = collator(data)

    assert len(batch) == 2  # The number of keys in the batch.
    assert torch.equal(batch[length_key], torch.tensor([4, 3, 2]))
    gt_feature = torch.tensor([4, 3, 2, 4, 3, 2, 4, 3, 4], dtype=torch.float)
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_packed_sequence_collator_2d():
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
    collator = PackedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_packed_sequence_collator_3d():
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
    collator = PackedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[2, 2, 2], [2, 2, 2]],
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[2, 2, 2], [2, 2, 2]],
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4]]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_packed_sequence_collator_remove_empty():
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
    collator = PackedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [4, 4],
            [3, 3],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([2, 2, 2, 1]))


def test_packed_sequence_collator_with_mask():
    data = [
        (
            {ct.LENGTH: 2, INDEX: 0, NAME: "item0"},
            {
                MASK: torch.ones(2, dtype=torch.long),
                FEATURE: torch.full((2, 2), 2, dtype=torch.float),
            },
        ),
        (
            {ct.LENGTH: 3, INDEX: 1, NAME: "item1"},
            {
                MASK: torch.ones(3, dtype=torch.long),
                FEATURE: torch.full((3, 2), 3, dtype=torch.float),
            },
        ),
        (
            {ct.LENGTH: 4, INDEX: 2, NAME: "item2"},
            {
                MASK: torch.ones(4, dtype=torch.long),
                FEATURE: torch.full((4, 2), 4, dtype=torch.float),
            },
        ),
    ]
    collator = PackedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 5  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[ct.LENGTH], torch.tensor([4, 3, 2]))
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))

    assert torch.equal(batch[MASK].data, torch.ones(9, dtype=torch.long))
    assert torch.equal(batch[MASK].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_packed_sequence_collator_batch_single_example():
    data = [({ct.LENGTH: 2}, {FEATURE: torch.full((2,), 2, dtype=torch.float)})]
    collator = PackedSequenceCollator()
    batch = collator(data)

    assert len(batch) == 2  # The number of keys in the batch.
    assert torch.equal(batch[ct.LENGTH], torch.tensor([2]))
    gt_feature = torch.tensor([2, 2], dtype=torch.float)
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([1, 1]))


######################################
#     DictPackedSequenceCollator     #
######################################


def test_dict_packed_sequence_collator_str():
    assert str(DictPackedSequenceCollator(["something"])).startswith("DictPackedSequenceCollator(")


@mark.parametrize("keys_to_pack", (("key", 1), ["key", 1]))
def test_dict_packed_sequence_collator_keys_to_pack(
    keys_to_pack: Union[list[Hashable], tuple[Hashable, ...]]
):
    assert DictPackedSequenceCollator(keys_to_pack)._keys_to_pack == ("key", 1)


def test_dict_packed_sequence_1d():
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE])
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    gt_feature = torch.tensor([4, 3, 2, 4, 3, 2, 4, 3, 4], dtype=torch.float)
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_dict_packed_sequence_2d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_dict_packed_sequence_3d():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((2, 2, 3), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2, 3), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2, 3), 4, dtype=torch.float)},
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[2, 2, 2], [2, 2, 2]],
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[2, 2, 2], [2, 2, 2]],
            [[4, 4, 4], [4, 4, 4]],
            [[3, 3, 3], [3, 3, 3]],
            [[4, 4, 4], [4, 4, 4]]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_dict_packed_sequence_remove_empty():
    data = [
        {INDEX: 0, NAME: "item0", FEATURE: torch.full((0, 2), 2, dtype=torch.float)},
        {INDEX: 1, NAME: "item1", FEATURE: torch.full((3, 2), 3, dtype=torch.float)},
        {INDEX: 2, NAME: "item2", FEATURE: torch.full((4, 2), 4, dtype=torch.float)},
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE])
    batch = collator(data)

    assert len(batch) == 3  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1"]
    assert torch.equal(batch[INDEX], torch.tensor([2, 1]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [4, 4],
            [3, 3],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([2, 2, 2, 1]))


def test_dict_packed_sequence_mask():
    data = [
        {
            INDEX: 0,
            NAME: "item0",
            MASK: torch.ones(2, dtype=torch.long),
            FEATURE: torch.full((2, 2), 2, dtype=torch.float),
        },
        {
            INDEX: 1,
            NAME: "item1",
            MASK: torch.ones(3, dtype=torch.long),
            FEATURE: torch.full((3, 2), 3, dtype=torch.float),
        },
        {
            INDEX: 2,
            NAME: "item2",
            MASK: torch.ones(4, dtype=torch.long),
            FEATURE: torch.full((4, 2), 4, dtype=torch.float),
        },
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE, MASK])
    batch = collator(data)

    assert len(batch) == 4  # The number of keys in the batch.
    assert batch[NAME] == ["item2", "item1", "item0"]
    assert torch.equal(batch[INDEX], torch.tensor([2, 1, 0]))
    # fmt: off
    gt_feature = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [2, 2],
            [4, 4],
            [3, 3],
            [4, 4]
        ],
        dtype=torch.float
    )
    # fmt: on
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))

    assert torch.equal(batch[MASK].data, torch.ones(9, dtype=torch.long))
    assert torch.equal(batch[MASK].batch_sizes, torch.tensor([3, 3, 2, 1]))


def test_dict_packed_sequence_batch_single_example():
    data = [{FEATURE: torch.full((2,), 2, dtype=torch.float)}]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE])
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    gt_feature = torch.tensor([2, 2], dtype=torch.float)
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([1, 1]))


def test_dict_packed_sequence_ignore_extra_key():
    data = [
        {FEATURE: torch.full((2,), 2, dtype=torch.float)},
        {FEATURE: torch.full((3,), 3, dtype=torch.float)},
        {FEATURE: torch.full((4,), 4, dtype=torch.float)},
    ]
    collator = DictPackedSequenceCollator(keys_to_pack=[FEATURE, INDEX])
    batch = collator(data)

    assert len(batch) == 1  # The number of keys in the batch.
    gt_feature = torch.tensor([4, 3, 2, 4, 3, 2, 4, 3, 4], dtype=torch.float)
    assert torch.equal(batch[FEATURE].data, gt_feature)
    assert torch.equal(batch[FEATURE].batch_sizes, torch.tensor([3, 3, 2, 1]))
