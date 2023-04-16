import math
from typing import Union

import torch
from coola import objects_are_allclose
from pytest import mark, raises
from torch import Tensor

from gravitorch.utils.data_summary import (
    EmptyDataSummaryError,
    FloatTensorDataSummary,
    FloatTensorSequenceDataSummary,
)

############################################
#     Tests for FloatTensorDataSummary     #
############################################


def test_float_tensor_data_summary_str() -> None:
    assert str(FloatTensorDataSummary()).startswith("FloatTensorDataSummary(")


def test_float_tensor_data_summary_add_1_tensor() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (0.0, 3.0, 1.0, 4.0)


def test_float_tensor_data_summary_add_2_tensors() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor(0, dtype=torch.float))
    summary.add(torch.tensor([3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (0.0, 3.0, 1.0, 4.0)


@mark.parametrize(
    "tensor",
    (
        torch.ones(3 * 4 * 5),
        torch.ones(3, 4 * 5),
        torch.ones(3 * 4, 5),
        torch.ones(3, 4, 5),
        torch.ones(3, 4, 5, dtype=torch.int),
        torch.ones(3, 4, 5, dtype=torch.long),
        torch.ones(3, 4, 5, dtype=torch.double),
    ),
)
def test_float_tensor_data_summary_add_tensor(tensor: Tensor) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert tuple(summary._values) == (1.0,) * 60


def test_float_tensor_data_summary_add_max_size_3() -> None:
    summary = FloatTensorDataSummary(max_size=3)
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert tuple(summary._values) == (3.0, 1.0, 4.0)


def test_float_tensor_data_summary_add_empty_tensor() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([]))
    assert tuple(summary._values) == ()


@mark.parametrize(
    "tensor,count", ((torch.ones(5), 5), (torch.tensor([4, 2]), 2), (torch.arange(11), 11))
)
def test_float_tensor_data_summary_count(tensor: Tensor, count: int) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.count() == count


def test_float_tensor_data_summary_count_empty() -> None:
    summary = FloatTensorDataSummary()
    assert summary.count() == 0


@mark.parametrize(
    "tensor,max_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 4.0), (torch.arange(11), 10.0)),
)
def test_float_tensor_data_summary_max(tensor: Tensor, max_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.max() == max_value


def test_float_tensor_data_summary_max_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.max()


@mark.parametrize(
    "tensor,mean_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 3.0), (torch.arange(11), 5.0)),
)
def test_float_tensor_data_summary_mean(tensor: Tensor, mean_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.mean() == mean_value


def test_float_tensor_data_summary_mean_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.mean()


@mark.parametrize(
    "tensor,median_value",
    ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 2.0), (torch.arange(11), 5.0)),
)
def test_float_tensor_data_summary_median(tensor: Tensor, median_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.median() == median_value


def test_float_tensor_data_summary_median_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.median()


@mark.parametrize(
    "tensor,min_value", ((torch.ones(5), 1.0), (torch.tensor([4, 2]), 2.0), (torch.arange(11), 0.0))
)
def test_float_tensor_data_summary_min(tensor: Tensor, min_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.min() == min_value


def test_float_tensor_data_summary_min_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.min()


def test_float_tensor_data_summary_quantiles_default_quantiles() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.arange(21))
    assert summary.quantiles().equal(
        torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    )


@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8), torch.tensor([0.2, 0.8]), [0.8, 0.2]))
def test_float_tensor_data_summary_quantiles_custom_quantiles(
    quantiles: Union[Tensor, tuple[float, ...], list[float]]
) -> None:
    summary = FloatTensorDataSummary(quantiles=quantiles)
    summary.add(torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float))
    assert summary.quantiles().equal(torch.tensor([1.0, 4.0]))


def test_float_tensor_data_summary_quantiles_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.quantiles()


@mark.parametrize("tensor,std_value", ((torch.ones(5), 0.0), (torch.tensor([-1, 1]), math.sqrt(2))))
def test_float_tensor_data_summary_std(tensor: Tensor, std_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert math.isclose(summary.std(), std_value, abs_tol=1e-6)


def test_float_tensor_data_summary_std_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.std()


@mark.parametrize(
    "tensor,sum_value",
    ((torch.ones(5), 5.0), (torch.tensor([4, 2]), 6.0), (torch.arange(11), 55.0)),
)
def test_float_tensor_data_summary_sum(tensor: Tensor, sum_value: float) -> None:
    summary = FloatTensorDataSummary()
    summary.add(tensor)
    assert summary.sum() == sum_value


def test_float_tensor_data_summary_sum_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.sum()


def test_float_tensor_data_summary_values_empty() -> None:
    summary = FloatTensorDataSummary()
    assert tuple(summary._values) == ()


def test_float_tensor_data_summary_reset() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    summary.reset()
    assert tuple(summary._values) == ()


def test_float_tensor_data_summary_summary_default_quantiles() -> None:
    summary = FloatTensorDataSummary()
    summary.add(torch.arange(21))
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 21,
            "sum": 210.0,
            "mean": 10.0,
            "median": 10.0,
            "std": 6.204836845397949,
            "max": 20.0,
            "min": 0.0,
            "quantile 0.000": 0.0,
            "quantile 0.100": 2.0,
            "quantile 0.200": 4.0,
            "quantile 0.300": 6.0,
            "quantile 0.400": 8.0,
            "quantile 0.500": 10.0,
            "quantile 0.600": 12.0,
            "quantile 0.700": 14.0,
            "quantile 0.800": 16.0,
            "quantile 0.900": 18.0,
            "quantile 1.000": 20.0,
        },
    )


@mark.parametrize("quantiles", ([], (), torch.tensor([])))
def test_float_tensor_data_summary_summary_default_no_quantile(
    quantiles: Union[Tensor, tuple[float, ...], list[float]]
) -> None:
    summary = FloatTensorDataSummary(quantiles=quantiles)
    summary.add(torch.ones(5))
    assert objects_are_allclose(
        summary.summary(),
        {
            "count": 5,
            "sum": 5.0,
            "mean": 1.0,
            "median": 1.0,
            "std": 0.0,
            "max": 1.0,
            "min": 1.0,
        },
    )


def test_float_tensor_data_summary_summary_empty() -> None:
    summary = FloatTensorDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()


####################################################
#     Tests for FloatTensorSequenceDataSummary     #
####################################################


def test_float_tensor_sequence_data_summary_str() -> None:
    assert str(FloatTensorSequenceDataSummary()).startswith("FloatTensorSequenceDataSummary(")


def test_float_tensor_sequence_data_summary_add() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    assert summary._value_summary.count() == 4
    assert summary._length_summary.count() == 1


def test_float_tensor_sequence_data_summary_reset() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    summary.reset()
    assert summary._value_summary.count() == 0
    assert summary._length_summary.count() == 0


def test_float_tensor_sequence_data_summary_summary() -> None:
    summary = FloatTensorSequenceDataSummary()
    summary.add(torch.tensor([0, 3, 1, 4], dtype=torch.float))
    stats = summary.summary()
    assert len(stats) == 2
    assert "value" in stats
    assert "length" in stats


def test_float_tensor_sequence_data_summary_summary_empty() -> None:
    summary = FloatTensorSequenceDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()
