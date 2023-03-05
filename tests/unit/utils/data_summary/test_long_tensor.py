import torch
from coola import objects_are_allclose
from pytest import mark, raises
from torch import Tensor

from gravitorch.utils.data_summary import (
    EmptyDataSummaryError,
    LongTensorDataSummary,
    LongTensorSequenceDataSummary,
)

###########################################
#     Tests for LongTensorDataSummary     #
###########################################


def test_long_tensor_data_summary_str() -> None:
    assert str(LongTensorDataSummary()).startswith("LongTensorDataSummary(")


def test_long_tensor_data_summary_add_1_tensor() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.long))
    assert summary.count() == 4


def test_long_tensor_data_summary_add_3_tensor() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0], dtype=torch.long))
    summary.add(torch.tensor([3, 1, 4], dtype=torch.long))
    assert summary.count() == 4


@mark.parametrize(
    "tensor",
    (
        torch.ones(3 * 4 * 5, dtype=torch.long),  # 1D tensor
        torch.ones(3, 4 * 5, dtype=torch.long),  # 2D tensor
        torch.ones(3 * 4, 5, dtype=torch.long),  # 2D tensor
        torch.ones(3, 4, 5, dtype=torch.long),  # 3D tensor
        torch.ones(3, 4, 5, dtype=torch.int),  # int tensor
        torch.ones(3, 4, 5, dtype=torch.float),  # float tensor
        torch.ones(3, 4, 5, dtype=torch.double),  # double tensor
    ),
)
def test_long_tensor_data_summary_add_tensor(tensor: Tensor) -> None:
    summary = LongTensorDataSummary()
    summary.add(tensor)
    assert summary.count() == 60


def test_long_tensor_data_summary_add_empty_tensor() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([]))
    assert summary.count() == 0


def test_long_tensor_data_summary_count_empty() -> None:
    summary = LongTensorDataSummary()
    assert summary.count() == 0


def test_long_tensor_data_summary_most_common() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0, 4, 1, 3, 0, 1, 0], dtype=torch.long))
    assert summary.most_common() == [(0, 3), (1, 2), (4, 1), (3, 1)]


def test_long_tensor_data_summary_most_common_2() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0, 4, 1, 3, 0, 1, 0], dtype=torch.long))
    assert summary.most_common(2) == [(0, 3), (1, 2)]


def test_long_tensor_data_summary_most_common_empty() -> None:
    summary = LongTensorDataSummary()
    with raises(EmptyDataSummaryError):
        summary.most_common()


def test_long_tensor_data_summary_reset() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.long))
    summary.reset()
    assert len(tuple(summary._counter.elements())) == 0


def test_long_tensor_data_summary_summary() -> None:
    summary = LongTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4, 0, 1, 0], dtype=torch.long))
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 7,
            "num_unique_values": 4,
            "count_0": 3,
            "count_1": 2,
            "count_3": 1,
            "count_4": 1,
        },
    )


def test_long_tensor_data_summary_summary_empty() -> None:
    summary = LongTensorDataSummary()
    with raises(EmptyDataSummaryError):
        summary.summary()


###################################################
#     Tests for LongTensorSequenceDataSummary     #
###################################################


def test_long_tensor_sequence_data_summary_str() -> None:
    assert str(LongTensorSequenceDataSummary()).startswith("LongTensorSequenceDataSummary(")


def test_long_tensor_sequence_data_summary_add() -> None:
    summary = LongTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.long))
    assert summary._value_summary.count() == 4
    assert summary._length_summary.count() == 1


def test_long_tensor_sequence_data_summary_reset() -> None:
    summary = LongTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.long))
    summary.reset()
    assert summary._value_summary.count() == 0
    assert summary._length_summary.count() == 0


def test_long_tensor_sequence_data_summary_summary() -> None:
    summary = LongTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4, 0, 1, 0], dtype=torch.long))
    stats = summary.summary()
    assert len(stats) == 2
    assert "value" in stats
    assert "length" in stats


def test_long_tensor_sequence_data_summary_summary_empty() -> None:
    summary = LongTensorSequenceDataSummary()
    with raises(EmptyDataSummaryError):
        summary.summary()
