from pytest import raises

from gravitorch.experimental.dataflow import IterableDataFlow

######################################
#     Tests for IterableDataFlow     #
######################################


def test_iterable_dataflow_str() -> None:
    assert str(IterableDataFlow([1, 2, 3, 4, 5])).startswith("IterableDataFlow(")


def test_iterable_dataflow_incorrect_type() -> None:
    with raises(TypeError):
        IterableDataFlow(1)


def test_iterable_dataflow_iter() -> None:
    with IterableDataFlow([1, 2, 3, 4, 5]) as flow:
        assert list(flow) == [1, 2, 3, 4, 5]
