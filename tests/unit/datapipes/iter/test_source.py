from collections.abc import Iterable
from unittest.mock import Mock

from pytest import mark, raises

from gravitorch.datapipes.iter import SourceWrapper

###################################
#     Tests for SourceWrapper     #
###################################


def test_source_wrapper_str() -> None:
    assert str(SourceWrapper([])).startswith("SourceWrapperIterDataPipe(")


@mark.parametrize("source", ([1, 2, 3], (1, 2, 3), (i for i in range(1, 4))))
def test_source_wrapper_iter(source: Iterable) -> None:
    assert list(SourceWrapper(source)) == [1, 2, 3]


def test_source_wrapper_iter_deepcopy_true() -> None:
    datapipe = SourceWrapper([[0, i] for i in range(1, 4)], deepcopy=True)
    for item in datapipe:
        item.append(2)
    assert list(datapipe) == [[0, 1], [0, 2], [0, 3]]


def test_source_wrapper_iter_deepcopy_false() -> None:
    datapipe = SourceWrapper([[0, i] for i in range(1, 4)])
    for item in datapipe:
        item.append(2)
    assert list(datapipe) == [[0, 1, 2], [0, 2, 2], [0, 3, 2]]


@mark.parametrize("deepcopy", (True, False))
def test_source_wrapper_iter_impossible_deepcopy(deepcopy: bool) -> None:
    datapipe = SourceWrapper(([0, i] for i in range(1, 4)), deepcopy=deepcopy)
    assert list(datapipe) == [[0, 1], [0, 2], [0, 3]]
    assert list(datapipe) == []


def test_source_wrapper_len() -> None:
    assert len(SourceWrapper(Mock(__len__=Mock(return_value=5)))) == 5


def test_source_wrapper_no_len() -> None:
    with raises(TypeError):
        len(SourceWrapper(Mock()))
