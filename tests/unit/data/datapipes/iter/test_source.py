from collections.abc import Iterable
from unittest.mock import Mock

from pytest import mark, raises

from gravitorch.data.datapipes.iter import SourceIterDataPipe

########################################
#     Tests for SourceIterDataPipe     #
########################################


def test_source_iter_datapipe_str():
    assert str(SourceIterDataPipe([])).startswith("SourceIterDataPipe(")


@mark.parametrize("data", ([1, 2, 3], (1, 2, 3), (i for i in range(1, 4))))
def test_source_iter_datapipe_iter(data: Iterable):
    assert list(SourceIterDataPipe(data)) == [1, 2, 3]


def test_source_iter_datapipe_iter_deepcopy_true():
    datapipe = SourceIterDataPipe([[0, i] for i in range(1, 4)], deepcopy=True)
    for item in datapipe:
        item.append(2)
    assert list(datapipe) == [[0, 1], [0, 2], [0, 3]]


def test_source_iter_datapipe_iter_deepcopy_false():
    datapipe = SourceIterDataPipe([[0, i] for i in range(1, 4)])
    for item in datapipe:
        item.append(2)
    assert list(datapipe) == [[0, 1, 2], [0, 2, 2], [0, 3, 2]]


@mark.parametrize("deepcopy", (True, False))
def test_source_iter_datapipe_iter_impossible_deepcopy(deepcopy: bool):
    datapipe = SourceIterDataPipe(([0, i] for i in range(1, 4)), deepcopy=deepcopy)
    assert list(datapipe) == [[0, 1], [0, 2], [0, 3]]
    assert list(datapipe) == []


def test_source_iter_datapipe_len():
    assert len(SourceIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_source_iter_datapipe_no_len():
    with raises(TypeError):
        len(SourceIterDataPipe(Mock()))
