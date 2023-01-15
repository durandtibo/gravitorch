from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import mark, raises

from gravitorch.data.datapipes.iter import SourceIterDataPipe
from gravitorch.data.datapipes.iter.experimental import (
    PartialTransposerIterDataPipe,
    TransposerIterDataPipe,
)

############################################
#     Tests for TransposerIterDataPipe     #
############################################


def test_transposer_iter_datapipe_str():
    assert str(TransposerIterDataPipe(SourceIterDataPipe([]), dim0=0, dim1=1)).startswith(
        "TransposerIterDataPipe("
    )


@mark.parametrize("dim0", (0, 2))
def test_transposer_iter_datapipe_dim0(dim0: int):
    assert TransposerIterDataPipe(SourceIterDataPipe([]), dim0=dim0, dim1=1).dim0 == dim0


@mark.parametrize("dim1", (1, 2))
def test_transposer_iter_datapipe_dim1(dim1: int):
    assert TransposerIterDataPipe(SourceIterDataPipe([]), dim0=0, dim1=dim1).dim1 == dim1


def test_transposer_iter_datapipe_iter():
    datapipe = TransposerIterDataPipe(
        SourceIterDataPipe([torch.ones(3, 2), torch.zeros(2, 4, 1)]), dim0=0, dim1=1
    )
    assert objects_are_equal(tuple(datapipe), (torch.ones(2, 3), torch.zeros(4, 2, 1)))


def test_transposer_iter_datapipe_len():
    assert len(TransposerIterDataPipe(Mock(__len__=Mock(return_value=5)), dim0=0, dim1=1)) == 5


def test_transposer_iter_datapipe_no_len():
    with raises(TypeError):
        len(TransposerIterDataPipe(SourceIterDataPipe(i for i in range(5)), dim0=0, dim1=1))


###################################################
#     Tests for PartialTransposerIterDataPipe     #
###################################################


def test_partial_transposer_iter_datapipe_str():
    assert str(PartialTransposerIterDataPipe(SourceIterDataPipe([]), dims={})).startswith(
        "PartialTransposerIterDataPipe("
    )


def test_partial_transposer_iter_datapipe_iter():
    datapipe = PartialTransposerIterDataPipe(
        SourceIterDataPipe(
            [
                {"a": torch.ones(3, 2), "b": torch.zeros(2, 4, 1)},
                {"a": torch.zeros(2, 3), "b": torch.ones(4, 2, 1)},
            ]
        ),
        dims={"a": (0, 1), "b": (1, 2)},
    )
    assert objects_are_equal(
        tuple(datapipe),
        (
            {"a": torch.ones(2, 3), "b": torch.zeros(2, 1, 4)},
            {"a": torch.zeros(3, 2), "b": torch.ones(4, 1, 2)},
        ),
    )


def test_partial_transposer_iter_datapipe_len():
    assert len(PartialTransposerIterDataPipe(Mock(__len__=Mock(return_value=5)), dims={})) == 5


def test_partial_transposer_iter_datapipe_no_len():
    with raises(TypeError):
        len(PartialTransposerIterDataPipe(SourceIterDataPipe(i for i in range(5)), dims={}))
