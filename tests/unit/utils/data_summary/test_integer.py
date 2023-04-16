from coola import objects_are_allclose
from pytest import raises

from gravitorch.utils.data_summary import EmptyDataSummaryError, IntegerDataSummary

########################################
#     Tests for IntegerDataSummary     #
########################################


def test_integer_data_summary_str() -> None:
    assert str(IntegerDataSummary()).startswith("IntegerDataSummary(")


def test_integer_data_summary_add_one_call() -> None:
    summary = IntegerDataSummary()
    summary.add(0)
    assert summary.count() == 1


def test_integer_data_summary_add_two_calls() -> None:
    summary = IntegerDataSummary()
    summary.add(0)
    summary.add(1)
    assert summary.count() == 2


def test_integer_data_summary_count_empty() -> None:
    summary = IntegerDataSummary()
    assert summary.count() == 0


def test_integer_data_summary_most_common() -> None:
    summary = IntegerDataSummary()
    for i in [0, 4, 1, 3, 0, 1, 0]:
        summary.add(i)
    assert summary.most_common() == [(0, 3), (1, 2), (4, 1), (3, 1)]


def test_integer_data_summary_most_common_2() -> None:
    summary = IntegerDataSummary()
    for i in [0, 4, 1, 3, 0, 1, 0]:
        summary.add(i)
    assert summary.most_common(2) == [(0, 3), (1, 2)]


def test_integer_data_summary_most_common_empty() -> None:
    summary = IntegerDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.most_common()


def test_integer_data_summary_reset() -> None:
    summary = IntegerDataSummary()
    for i in [0, 3, 1, 4]:
        summary.add(i)
    summary.reset()
    assert len(tuple(summary._counter.elements())) == 0


def test_integer_data_summary_summary() -> None:
    summary = IntegerDataSummary()
    for i in [0, 3, 1, 4, 0, 1, 0]:
        summary.add(i)
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


def test_integer_data_summary_summary_empty() -> None:
    summary = IntegerDataSummary()
    with raises(EmptyDataSummaryError, match="The summary is empty"):
        summary.summary()
