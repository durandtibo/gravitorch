from unittest.mock import Mock

import torch
from coola import objects_are_allclose, objects_are_equal
from pytest import raises

from gravitorch.data.datapipes.iter import SourceIterDataPipe
from gravitorch.data.datapipes.iter.experimental import (
    ClampTensorIterDataPipe,
    ContiguousTensorIterDataPipe,
    SymlogTensorIterDataPipe,
)

#############################################
#     Tests for ClampTensorIterDataPipe     #
#############################################


def test_clamp_tensor_iter_datapipe_str():
    assert str(ClampTensorIterDataPipe(SourceIterDataPipe([]), key="one", min=0)).startswith(
        "ClampTensorIterDataPipe("
    )


def test_clamp_tensor_iter_datapipe_min_max_none():
    with raises(ValueError):
        ClampTensorIterDataPipe(SourceIterDataPipe([]), key="one")


def test_clamp_tensor_iter_datapipe_iter_min_0():
    assert objects_are_equal(
        list(
            ClampTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", min=0
            )
        ),
        [{"one": torch.tensor([0, 0, 0, 1, 2, 3, 4, 5], dtype=torch.float)}],
    )


def test_clamp_tensor_iter_datapipe_iter_max_2():
    assert objects_are_equal(
        list(
            ClampTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", max=2
            )
        ),
        [{"one": torch.tensor([-2, -1, 0, 1, 2, 2, 2, 2], dtype=torch.float)}],
    )


def test_clamp_tensor_iter_datapipe_iter_min_0_max_2():
    assert objects_are_equal(
        list(
            ClampTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", min=0, max=2
            )
        ),
        [{"one": torch.tensor([0, 0, 0, 1, 2, 2, 2, 2], dtype=torch.float)}],
    )


def test_clamp_tensor_iter_datapipe_len():
    assert len(ClampTensorIterDataPipe(Mock(__len__=Mock(return_value=5)), key="one", min=0)) == 5


def test_clamp_tensor_iter_datapipe_no_len():
    with raises(TypeError):
        len(ClampTensorIterDataPipe(SourceIterDataPipe(i for i in range(5)), key="one", min=0))


##################################################
#     Tests for ContiguousTensorIterDataPipe     #
##################################################


def test_contiguous_tensor_iter_datapipe_str():
    assert str(ContiguousTensorIterDataPipe(SourceIterDataPipe([]))).startswith(
        "ContiguousTensorIterDataPipe("
    )


def test_contiguous_tensor_iter_datapipe_iter():
    source = SourceIterDataPipe(
        [torch.ones(3, 2).transpose(0, 1), torch.zeros(2, 4).transpose(0, 1)]
    )
    assert [tensor.is_contiguous() for tensor in ContiguousTensorIterDataPipe(source)] == [
        True,
        True,
    ]


def test_contiguous_tensor_iter_datapipe_len():
    assert len(ContiguousTensorIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_contiguous_tensor_iter_datapipe_no_len():
    with raises(TypeError):
        len(ContiguousTensorIterDataPipe(SourceIterDataPipe(i for i in range(5))))


####################################################
#     Tests for SymlogTensorIterDataPipe     #
####################################################


def test_symlog_tensor_iter_datapipe_iter_min_0():
    assert objects_are_allclose(
        list(
            SymlogTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", min=0
            )
        ),
        [
            {
                "one": torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.6931471824645996,
                        1.0986123085021973,
                        1.3862943649291992,
                        1.6094379425048828,
                        1.7917594909667969,
                    ],
                    dtype=torch.float,
                )
            }
        ],
    )


def test_symlog_tensor_iter_datapipe_iter_max_2():
    assert objects_are_allclose(
        list(
            SymlogTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", max=2
            )
        ),
        [
            {
                "one": torch.tensor(
                    [
                        -1.0986123085021973,
                        -0.6931471824645996,
                        0.0,
                        0.6931471824645996,
                        1.0986123085021973,
                        1.0986123085021973,
                        1.0986123085021973,
                        1.0986123085021973,
                    ],
                    dtype=torch.float,
                )
            }
        ],
    )


def test_symlog_tensor_iter_datapipe_iter_min_0_max_2():
    assert objects_are_allclose(
        list(
            SymlogTensorIterDataPipe(
                SourceIterDataPipe([{"one": torch.linspace(-2, 5, 8)}]), key="one", min=0, max=2
            )
        ),
        [
            {
                "one": torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.6931471824645996,
                        1.0986123085021973,
                        1.0986123085021973,
                        1.0986123085021973,
                        1.0986123085021973,
                    ],
                    dtype=torch.float,
                )
            }
        ],
    )
