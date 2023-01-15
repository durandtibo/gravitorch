from pytest import mark, raises

from gravitorch.data.datapipes.iter import FixedLengthIterDataPipe, SourceIterDataPipe

#############################################
#     Tests for FixedLengthIterDataPipe     #
#############################################


def test_fixed_length_iter_datapipe_str():
    assert str(FixedLengthIterDataPipe(SourceIterDataPipe([]), length=42)).startswith(
        "FixedLengthIterDataPipe("
    )


def test_fixed_length_iter_datapipe_iter_exact_length():
    datapipe = FixedLengthIterDataPipe(SourceIterDataPipe([1, 2, 3]), length=3)
    assert tuple(datapipe) == (1, 2, 3)


def test_fixed_length_iter_datapipe_iter_too_long():
    datapipe = FixedLengthIterDataPipe(SourceIterDataPipe([1, 2, 3, 4, 5]), length=3)
    assert tuple(datapipe) == (1, 2, 3)


def test_fixed_length_iter_datapipe_iter_too_short():
    datapipe = FixedLengthIterDataPipe(SourceIterDataPipe([1, 2, 3]), length=5)
    assert tuple(datapipe) == (1, 2, 3, 1, 2)


@mark.parametrize("length", (1, 2, 3))
def test_fixed_length_iter_datapipe_len(length: int):
    assert (
        len(FixedLengthIterDataPipe(SourceIterDataPipe([1, 2, 3, 4, 5]), length=length)) == length
    )


@mark.parametrize("length", (-1, 0))
def test_fixed_length_iter_datapipe_incorrect_length(length: int):
    with raises(ValueError):
        FixedLengthIterDataPipe(SourceIterDataPipe([1, 2, 3, 4, 5]), length=length)
