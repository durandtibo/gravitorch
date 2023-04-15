from pytest import mark, raises

from gravitorch.datapipes.iter import Looper, SourceWrapper

############################
#     Tests for Looper     #
############################


def test_looper_str() -> None:
    assert str(Looper(SourceWrapper([]), length=42)).startswith("LooperIterDataPipe(")


def test_looper_iter_exact_length() -> None:
    assert tuple(Looper(SourceWrapper([1, 2, 3]), length=3)) == (1, 2, 3)


def test_looper_iter_too_long() -> None:
    assert tuple(Looper(SourceWrapper([1, 2, 3, 4, 5]), length=3)) == (1, 2, 3)


def test_looper_iter_too_short() -> None:
    assert tuple(Looper(SourceWrapper([1, 2, 3]), length=5)) == (1, 2, 3, 1, 2)


@mark.parametrize("length", (1, 2, 3))
def test_looper_len(length: int) -> None:
    assert len(Looper(SourceWrapper([1, 2, 3, 4, 5]), length=length)) == length


@mark.parametrize("length", (-1, 0))
def test_looper_incorrect_length(length: int) -> None:
    with raises(ValueError, match="Incorrect length:"):
        Looper(SourceWrapper([1, 2, 3, 4, 5]), length=length)
