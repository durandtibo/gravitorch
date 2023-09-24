from pytest import raises

from gravitorch.dataflows import IterableDataFlow

######################################
#     Tests for IterableDataFlow     #
######################################


def test_iterable_dataflow_str_with_length() -> None:
    assert str(IterableDataFlow([1, 2, 3, 4, 5])) == "IterableDataFlow(length=5)"


def test_iterable_dataflow_str_without_length() -> None:
    assert str(IterableDataFlow(i for i in range(5))) == "IterableDataFlow()"


def test_iterable_dataflow_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect type. Expecting iterable but received"):
        IterableDataFlow(1)


def test_iterable_dataflow_iter() -> None:
    with IterableDataFlow([1, 2, 3, 4, 5]) as flow:
        assert list(flow) == [1, 2, 3, 4, 5]


def test_iterable_dataflow_iter_deepcopy_true() -> None:
    dataflow = IterableDataFlow([[1, 2, 3], [4, 5, 6], [7, 8], [9]], deepcopy=True)
    with dataflow as flow:
        for batch in flow:
            batch.append(0)
    with dataflow as flow:
        assert list(flow) == [[1, 2, 3], [4, 5, 6], [7, 8], [9]]


def test_iterable_dataflow_iter_deepcopy_false() -> None:
    dataflow = IterableDataFlow([[1, 2, 3], [4, 5, 6], [7, 8], [9]])
    with dataflow as flow:
        for batch in flow:
            batch.append(0)
    with dataflow as flow:
        assert list(flow) == [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 0], [9, 0]]


def test_iterable_dataflow_iter_impossible_deepcopy() -> None:
    dataflow = IterableDataFlow((i for i in range(5)), deepcopy=True)
    with dataflow as flow:
        assert list(flow) == [0, 1, 2, 3, 4]
    with dataflow as flow:
        assert list(flow) == []


def test_iterable_dataflow_len() -> None:
    dataflow = IterableDataFlow([1, 2, 3, 4, 5])
    assert len(dataflow) == 5
    with dataflow as flow:
        assert len(flow) == 5


def test_iterable_dataflow_no_len() -> None:
    dataflow = IterableDataFlow(i for i in range(5))
    with raises(TypeError):
        len(dataflow)
    with dataflow as flow:
        with raises(TypeError):
            len(flow)
