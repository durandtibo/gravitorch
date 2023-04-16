import math
from typing import Union

import torch
from coola import objects_are_allclose
from pytest import mark, raises
from torch import Tensor

from gravitorch.utils.data_summary import EmptyDataSummaryError, FloatDataSummary

######################################
#     Tests for FloatDataSummary     #
######################################


def test_float_data_summary_str() -> None:
    assert str(FloatDataSummary()).startswith("FloatDataSummary(")


def test_float_data_summary_add_one_call() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    assert tuple(summary._values) == (0.0,)


def test_float_data_summary_add_two_calls() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert tuple(summary._values) == (3.0, 1.0)


def test_float_data_summary_count_1() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    assert summary.count() == 1


def test_float_data_summary_count_2() -> None:
    summary = FloatDataSummary()
    summary.add(0.0)
    summary.add(0.0)
    assert summary.count() == 2


def test_float_data_summary_count_empty() -> None:
    summary = FloatDataSummary()
    assert summary.count() == 0


def test_float_data_summary_max() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.max() == 3.0


def test_float_data_summary_max_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.max()


def test_float_data_summary_mean() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.mean() == 2.0


def test_float_data_summary_mean_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.mean()


def test_float_data_summary_median() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.median() == 1.0


def test_float_data_summary_median_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.median()


def test_float_data_summary_min() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.min() == 1.0


def test_float_data_summary_min_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.min()


def test_float_data_summary_quantiles_default_quantiles() -> None:
    summary = FloatDataSummary()
    for i in range(21):
        summary.add(i)
    assert summary.quantiles().equal(
        torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    )


@mark.parametrize("quantiles", ([0.2, 0.8], (0.2, 0.8), torch.tensor([0.2, 0.8]), [0.8, 0.2]))
def test_float_data_summary_quantiles_custom_quantiles(
    quantiles: Union[Tensor, tuple[float, ...], list[float]]
) -> None:
    summary = FloatDataSummary(quantiles=quantiles)
    for i in range(6):
        summary.add(i)
    assert summary.quantiles().equal(torch.tensor([1.0, 4.0]))


def test_float_data_summary_quantiles_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.quantiles()


def test_float_data_summary_std_1_value() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    assert math.isnan(summary.std())


def test_float_data_summary_std_0() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.add(1.0)
    assert summary.std() == 0.0


def test_float_data_summary_std_2() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.add(-1.0)
    assert math.isclose(summary.std(), math.sqrt(2), abs_tol=1e-6)


def test_float_data_summary_std_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.std()


def test_float_data_summary_sum() -> None:
    summary = FloatDataSummary()
    summary.add(3.0)
    summary.add(1.0)
    assert summary.sum() == 4.0


def test_float_data_summary_sum_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.sum()


def test_float_data_summary_values_empty() -> None:
    summary = FloatDataSummary()
    assert tuple(summary._values) == ()


def test_float_data_summary_reset() -> None:
    summary = FloatDataSummary()
    summary.add(1.0)
    summary.reset()
    assert tuple(summary._values) == ()


def test_float_data_summary_summary_default_quantiles() -> None:
    summary = FloatDataSummary()
    for i in range(21):
        summary.add(i)
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
def test_float_data_summary_summary_default_no_quantile(
    quantiles: Union[Tensor, tuple[float, ...], list[float]]
) -> None:
    summary = FloatDataSummary(quantiles=quantiles)
    for _ in range(5):
        summary.add(1.0)
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


def test_float_data_summary_summary_empty() -> None:
    summary = FloatDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()
