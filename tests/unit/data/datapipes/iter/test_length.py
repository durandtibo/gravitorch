from pytest import mark, raises

from gravitorch.data.datapipes.iter import Looper, SourceIterDataPipe

############################
#     Tests for Looper     #
############################


def test_looper_str():
    assert str(Looper(SourceIterDataPipe([]), length=42)).startswith("LooperIterDataPipe(")


def test_looper_iter_exact_length():
    assert tuple(Looper(SourceIterDataPipe([1, 2, 3]), length=3)) == (1, 2, 3)


def test_looper_iter_too_long():
    assert tuple(Looper(SourceIterDataPipe([1, 2, 3, 4, 5]), length=3)) == (1, 2, 3)


def test_looper_iter_too_short():
    assert tuple(Looper(SourceIterDataPipe([1, 2, 3]), length=5)) == (1, 2, 3, 1, 2)


@mark.parametrize("length", (1, 2, 3))
def test_looper_len(length: int):
    assert len(Looper(SourceIterDataPipe([1, 2, 3, 4, 5]), length=length)) == length


@mark.parametrize("length", (-1, 0))
def test_looper_incorrect_length(length: int):
    with raises(ValueError):
        Looper(SourceIterDataPipe([1, 2, 3, 4, 5]), length=length)
