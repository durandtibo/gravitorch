import torch
from pytest import mark

from gravitorch.utils.tensor import (
    isymlog,
    isymlog_,
    safeexp,
    safelog,
    scalable_quantile,
    symlog,
    symlog_,
)

DTYPES = (torch.float, torch.long)


#######################################
#     Tests for scalable_quantile     #
#######################################


@mark.parametrize("dtype", DTYPES)
def test_scalable_quantile_dtype(dtype: torch.dtype):
    assert scalable_quantile(torch.arange(11).to(dtype=dtype), q=torch.tensor([0.1])).equal(
        torch.tensor([1], dtype=torch.float),
    )


def test_scalable_quantile_q_multiple():
    assert scalable_quantile(torch.arange(11), q=torch.tensor([0.1, 0.5, 0.9])).equal(
        torch.tensor([1, 5, 9], dtype=torch.float),
    )


#############################
#     Tests for safeexp     #
#############################


def test_safeexp_max_value_default():
    assert safeexp(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float)).allclose(
        torch.tensor([0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 485165184.0]),
    )


def test_safeexp_max_value_1():
    assert safeexp(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float), max_value=1.0).equal(
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 2.7182817459106445, 2.7182817459106445]
        ),
    )


#############################
#     Tests for safelog     #
#############################


def test_safelog_min_value_default():
    assert safelog(torch.tensor([-1, 0, 1, 2], dtype=torch.float)).allclose(
        torch.tensor([-18.420680743952367, -18.420680743952367, 0.0, 0.6931471805599453]),
    )


def test_safelog_min_value_1():
    assert safelog(torch.tensor([-1, 0, 1, 2], dtype=torch.float), min_value=1.0).equal(
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453]),
    )


############################
#     Tests for symlog     #
############################


def test_symlog():
    assert symlog(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)).allclose(
        torch.tensor(
            [-1.0986122886681098, -0.6931471805599453, 0.0, 0.6931471805599453, 1.0986122886681098]
        ),
    )


def test_symlog_():
    tensor = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
    symlog_(tensor)
    assert tensor.allclose(
        torch.tensor(
            [-1.0986122886681098, -0.6931471805599453, 0.0, 0.6931471805599453, 1.0986122886681098]
        ),
    )


#############################
#     Tests for isymlog     #
#############################


def test_isymlog():
    assert isymlog(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)).allclose(
        torch.tensor(
            [-6.38905609893065, -1.718281828459045, 0, 1.718281828459045, 6.38905609893065],
            dtype=torch.float,
        ),
    )


def test_isymlog_cycle():
    assert isymlog(symlog(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float))).allclose(
        torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float),
    )


def test_isymlog_():
    tensor = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float)
    isymlog_(tensor)
    assert tensor.allclose(
        torch.tensor(
            [-6.38905609893065, -1.718281828459045, 0, 1.718281828459045, 6.38905609893065],
            dtype=torch.float,
        ),
    )
