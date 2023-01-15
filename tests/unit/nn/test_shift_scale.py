from collections.abc import Callable
from functools import partial

import torch
from pytest import mark

from gravitorch.nn.shift_scale import (
    SequenceShiftScale,
    ShiftScale,
    sequence_shift_scale,
    shift_scale,
)
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#################################
#     Tests for shift_scale     #
#################################


@mark.parametrize("module", (shift_scale, ShiftScale()))
def test_shift_scale_change_scale(module: Callable):
    # 1 example: src range [0, 10] -> dst range [0, 1]
    # 1 example: src range [0, 1] -> dst range [0, 2]
    out = module(
        torch.tensor([[0, 5, 10], [0, 0.5, 1]], dtype=torch.float),
        src_range=torch.tensor(
            [[[0, 0, 0], [10, 10, 10]], [[0, 0, 0], [1, 1, 1]]], dtype=torch.float
        ),
        dst_range=torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float),
    )
    assert out.equal(torch.tensor([[0, 0.5, 1], [0, 1, 2]], dtype=torch.float))


@mark.parametrize("module", (shift_scale, ShiftScale()))
def test_shift_scale_per_dimension(module: Callable):
    src_range = torch.tensor(
        [[[-1, -2, -4], [1, 2, 4]], [[-1, -2, -4], [0, 0, 0]]], dtype=torch.float
    )
    dst_range = torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float)
    # min range value
    assert module(
        torch.tensor([[-1, -2, -4], [-1, -2, -4]], dtype=torch.float), src_range, dst_range
    ).equal(torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float))
    # mid range value
    assert module(
        torch.tensor([[0, 0, 0], [-0.5, -1, -2]], dtype=torch.float), src_range, dst_range
    ).equal(torch.tensor([[0.5, 0.5, 0.5], [1, 1, 1]], dtype=torch.float))
    # max range value
    assert module(
        torch.tensor([[1, 2, 4], [0, 0, 0]], dtype=torch.float), src_range, dst_range
    ).equal(torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float))
    # one
    assert module(torch.ones(2, 3), src_range, dst_range).equal(
        torch.tensor([[1, 0.75, 0.625], [4, 3, 2.5]], dtype=torch.float)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_shift_scale_forward(device: str, batch_size: int, feature_size: int):
    device = torch.device(device)
    module = ShiftScale().to(device=device)
    src_range = torch.ones(batch_size, 2, feature_size)
    src_range[:, 0] = -src_range[:, 0]
    dst_range = torch.ones(batch_size, 2, feature_size)
    dst_range[:, 0] = 0
    out = module(torch.zeros(batch_size, feature_size), src_range, dst_range)
    assert out.equal(0.5 * torch.ones(batch_size, feature_size))


##########################################
#     Tests for sequence_shift_scale     #
##########################################


def test_sequence_shift_scale_str():
    assert str(SequenceShiftScale()).startswith("SequenceShiftScale(")


@mark.parametrize("batch_first", (True, False))
def test_sequence_shift_scale_batch_first(batch_first: bool):
    assert SequenceShiftScale(batch_first).batch_first == batch_first


@mark.parametrize(
    "module",
    (partial(sequence_shift_scale, batch_first=True), SequenceShiftScale(batch_first=True)),
)
def test_sequence_shift_scale_change_scale_batch_first(module: Callable):
    # 1 example: src range [0, 10] -> dst range [0, 1]
    # 1 example: src range [0, 1] -> dst range [0, 2]
    out = module(
        torch.tensor([[[0, 5, 10], [0, 1, 2]], [[0, 0.5, 1], [2, 1, 0]]], dtype=torch.float),
        src_range=torch.tensor(
            [[[0, 0, 0], [10, 10, 10]], [[0, 0, 0], [1, 1, 1]]], dtype=torch.float
        ),
        dst_range=torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float),
    )
    assert out.equal(
        torch.tensor([[[0, 0.5, 1], [0, 0.1, 0.2]], [[0, 1, 2], [4, 2, 0]]], dtype=torch.float)
    )


@mark.parametrize("module", (sequence_shift_scale, SequenceShiftScale()))
def test_sequence_shift_scale_change_scale_sequence_first(module: Callable):
    # 1 example: src range [0, 10] -> dst range [0, 1]
    # 1 example: src range [0, 1] -> dst range [0, 2]
    out = module(
        torch.tensor([[[0, 5, 10], [0, 0.5, 1]], [[0, 1, 2], [2, 1, 0]]], dtype=torch.float),
        src_range=torch.tensor(
            [[[0, 0, 0], [10, 10, 10]], [[0, 0, 0], [1, 1, 1]]], dtype=torch.float
        ),
        dst_range=torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float),
    )
    assert out.equal(
        torch.tensor([[[0, 0.5, 1], [0, 1, 2]], [[0, 0.1, 0.2], [4, 2, 0]]], dtype=torch.float)
    )


@mark.parametrize(
    "module",
    (partial(sequence_shift_scale, batch_first=True), SequenceShiftScale(batch_first=True)),
)
def test_sequence_shift_scale_per_dimension_batch_first(module: Callable):
    out = module(
        torch.tensor(
            [
                [[-1, -2, -4], [0, 0, 0], [1, 2, 4], [1, 1, 1]],
                [[-1, -2, -4], [-0.5, -1, -2], [0, 0, 0], [1, 1, 1]],
            ],
            dtype=torch.float,
        ),
        src_range=torch.tensor(
            [[[-1, -2, -4], [1, 2, 4]], [[-1, -2, -4], [0, 0, 0]]], dtype=torch.float
        ),
        dst_range=torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float),
    )
    assert out.equal(
        torch.tensor(
            [
                [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1], [1, 0.75, 0.625]],
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 3, 2.5]],
            ],
            dtype=torch.float,
        )
    )


@mark.parametrize("module", (sequence_shift_scale, SequenceShiftScale()))
def test_sequence_shift_scale_per_dimension_sequence_first(module: Callable):
    out = module(
        torch.tensor(
            [
                [[-1, -2, -4], [-1, -2, -4]],  # min range value
                [[0, 0, 0], [-0.5, -1, -2]],  # mid range value
                [[1, 2, 4], [0, 0, 0]],  # max range value
                torch.ones(2, 3),  # one
            ],
            dtype=torch.float,
        ),
        src_range=torch.tensor(
            [[[-1, -2, -4], [1, 2, 4]], [[-1, -2, -4], [0, 0, 0]]], dtype=torch.float
        ),
        dst_range=torch.tensor([[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]], dtype=torch.float),
    )
    assert out.equal(
        torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0]],  # min range value
                [[0.5, 0.5, 0.5], [1, 1, 1]],  # mid range value
                [[1, 1, 1], [2, 2, 2]],  # max range value
                [[1, 0.75, 0.625], [4, 3, 2.5]],  # one
            ],
            dtype=torch.float,
        )
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_sequence_shift_scale_forward_batch_first(
    device: str, batch_size: int, seq_len: int, feature_size: int
):
    device = torch.device(device)
    module = SequenceShiftScale(batch_first=True).to(device=device)
    src_range = torch.ones(batch_size, 2, feature_size, device=device)
    src_range[:, 0] = -src_range[:, 0]
    dst_range = torch.ones(batch_size, 2, feature_size, device=device)
    dst_range[:, 0] = 0
    out = module(
        torch.zeros(batch_size, seq_len, feature_size, device=device), src_range, dst_range
    )
    assert out.equal(0.5 * torch.ones(batch_size, seq_len, feature_size, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_sequence_shift_scale_forward_sequence_first(
    device: str, batch_size: int, seq_len: int, feature_size: int
):
    device = torch.device(device)
    module = SequenceShiftScale().to(device=device)
    src_range = torch.ones(batch_size, 2, feature_size, device=device)
    src_range[:, 0] = -src_range[:, 0]
    dst_range = torch.ones(batch_size, 2, feature_size, device=device)
    dst_range[:, 0] = 0
    out = module(
        torch.zeros(seq_len, batch_size, feature_size, device=device), src_range, dst_range
    )
    assert out.equal(0.5 * torch.ones(seq_len, batch_size, feature_size, device=device))
