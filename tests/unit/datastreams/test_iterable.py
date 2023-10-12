from pytest import raises

from gravitorch.datastreams import IterableDataStream

######################################
#     Tests for IterableDataStream     #
######################################


def test_iterable_datastream_str_with_length() -> None:
    assert str(IterableDataStream([1, 2, 3, 4, 5])) == "IterableDataStream(length=5)"


def test_iterable_datastream_str_without_length() -> None:
    assert str(IterableDataStream(i for i in range(5))) == "IterableDataStream()"


def test_iterable_datastream_incorrect_type() -> None:
    with raises(TypeError, match="Incorrect type. Expecting iterable but received"):
        IterableDataStream(1)


def test_iterable_datastream_iter() -> None:
    with IterableDataStream([1, 2, 3, 4, 5]) as flow:
        assert list(flow) == [1, 2, 3, 4, 5]


def test_iterable_datastream_iter_deepcopy_true() -> None:
    datastream = IterableDataStream([[1, 2, 3], [4, 5, 6], [7, 8], [9]], deepcopy=True)
    with datastream as flow:
        for batch in flow:
            batch.append(0)
    with datastream as flow:
        assert list(flow) == [[1, 2, 3], [4, 5, 6], [7, 8], [9]]


def test_iterable_datastream_iter_deepcopy_false() -> None:
    datastream = IterableDataStream([[1, 2, 3], [4, 5, 6], [7, 8], [9]])
    with datastream as flow:
        for batch in flow:
            batch.append(0)
    with datastream as flow:
        assert list(flow) == [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 0], [9, 0]]


def test_iterable_datastream_iter_impossible_deepcopy() -> None:
    datastream = IterableDataStream((i for i in range(5)), deepcopy=True)
    with datastream as flow:
        assert list(flow) == [0, 1, 2, 3, 4]
    with datastream as flow:
        assert list(flow) == []


def test_iterable_datastream_len() -> None:
    datastream = IterableDataStream([1, 2, 3, 4, 5])
    assert len(datastream) == 5
    with datastream as flow:
        assert len(flow) == 5


def test_iterable_datastream_no_len() -> None:
    datastream = IterableDataStream(i for i in range(5))
    with raises(TypeError):
        len(datastream)
    with datastream as flow:
        with raises(TypeError):
            len(flow)
