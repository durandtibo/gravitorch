import math

import torch
from pytest import mark
from torch import nn
from torch.optim import SGD

from gravitorch.nn import (
    Asinh,
    Clamp,
    Isymlog,
    Log1p,
    Mul,
    OnePolynomial,
    Safeexp,
    Safelog,
    SequenceToBatch,
    Sinh,
    Squeeze,
    Symlog,
    ToBinaryLabel,
    ToCategoricalLabel,
    ToFloat,
    ToLong,
)
from gravitorch.nn.utils import is_loss_decreasing
from gravitorch.utils import get_available_devices

DTYPES = (torch.long, torch.float)


###########################
#     Tests for Asinh     #
###########################


@mark.parametrize("device", get_available_devices())
def test_asinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Asinh().to(device=device)
    assert module(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [-1.4436354637145996, -0.8813735842704773, 0.0, 0.8813735842704773, 1.4436354637145996],
            dtype=torch.float,
            device=device,
        ),
    )


##########################
#     Tests for Sinh     #
##########################


@mark.parametrize("device", get_available_devices())
def test_sinh_forward(device: str) -> None:
    device = torch.device(device)
    module = Sinh().to(device=device)
    assert module(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [-3.6268603801727295, -1.175201177597046, 0.0, 1.175201177597046, 3.6268603801727295],
            dtype=torch.float,
            device=device,
        ),
    )


###########################
#     Tests for Log1p     #
###########################


@mark.parametrize("device", get_available_devices())
def test_log1p_forward(device: str) -> None:
    device = torch.device(device)
    module = Log1p().to(device=device)
    assert module(torch.tensor([0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [0.0, 0.6931471805599453, 1.0986122886681098], dtype=torch.float, device=device
        ),
    )


#########################
#     Tests for Mul     #
#########################


def test_mul_str() -> None:
    assert str(Mul(2.0)).startswith("Mul(")


@mark.parametrize("device", get_available_devices())
def test_mul_forward_2(device: str) -> None:
    device = torch.device(device)
    module = Mul(2.0).to(device=device)
    assert module(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([0, 2.0, 4.0], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_mul_forward_4(device: str) -> None:
    device = torch.device(device)
    module = Mul(4.0).to(device=device)
    assert module(torch.tensor([0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([0, 4.0, 8.0], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_mul_forward_long(device: str) -> None:
    device = torch.device(device)
    module = Mul(2.0).to(device=device)
    assert module(torch.tensor([0, 1, 2], dtype=torch.long, device=device)).allclose(
        torch.tensor([0, 2.0, 4.0], dtype=torch.float, device=device),
    )


###################################
#     Tests for OnePolynomial     #
###################################


def test_one_polynomial_str() -> None:
    assert str(OnePolynomial()).startswith("OnePolynomial(")


@mark.parametrize("device", get_available_devices())
def test_one_polynomial_forward(device: str) -> None:
    device = torch.device(device)
    module = OnePolynomial().to(device=device)
    assert module(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_one_polynomial_forward_alpha_2(device: str) -> None:
    device = torch.device(device)
    module = OnePolynomial(alpha=2).to(device=device)
    assert module(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([-2.0, 0.0, 2.0, 4.0], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_one_polynomial_forward_beta_1(device: str) -> None:
    device = torch.device(device)
    module = OnePolynomial(beta=1).to(device=device)
    assert module(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_one_polynomial_forward_gamma_2(device: str) -> None:
    device = torch.device(device)
    module = OnePolynomial(gamma=2).to(device=device)
    assert module(torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float, device=device)).allclose(
        torch.tensor([1.0, 0.0, 1.0, 4.0], dtype=torch.float, device=device),
    )


@mark.parametrize("alpha", (-1.0, 1.0, 2.0))
@mark.parametrize("beta", (-1.0, 0.0, 1.0))
@mark.parametrize("gamma", (0.5, 1.0))
def test_one_polynomial_is_loss_decreasing(alpha: float, beta: float, gamma: float) -> None:
    module = nn.Sequential(
        nn.Linear(4, 6),
        Clamp(min_value=0.0, max_value=1.0),
        OnePolynomial(alpha=alpha, beta=beta, gamma=gamma),
    )
    assert is_loss_decreasing(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.randn(10, 4),
        target=torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=torch.long),
        num_iterations=5,
    )


@mark.parametrize("gamma", (-1.0, 0.1, 0.5, 1.0))
def test_one_polynomial_is_loss_decreasing_range(gamma: float) -> None:
    module = nn.Sequential(
        nn.Linear(4, 6),
        Clamp(min_value=0.1, max_value=1.0),
        OnePolynomial.create_from_range(
            gamma=gamma,
            input_max_value=0.1,
            input_min_value=1.0,
            output_min_value=-1.0,
            output_max_value=1.0,
        ),
    )
    assert is_loss_decreasing(
        module=module,
        criterion=nn.CrossEntropyLoss(),
        optimizer=SGD(module.parameters(), lr=0.01),
        feature=torch.randn(10, 4),
        target=torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=torch.long),
        num_iterations=5,
    )


def test_one_polynomial_create_from_range() -> None:
    module = OnePolynomial.create_from_range(
        gamma=1.0,
        input_min_value=0.1,
        input_max_value=1.0,
        output_min_value=-1.0,
        output_max_value=1.0,
    )
    assert math.isclose(module._alpha, 2.2222222222222223)
    assert math.isclose(module._beta, -1.2222222222222223)


#############################
#     Tests for Safeexp     #
#############################


def test_safeexp_str() -> None:
    assert str(Safeexp()).startswith("Safeexp(")


@mark.parametrize("max_value", (5.0, 10.0))
def test_safeexp_max_value(max_value: float) -> None:
    assert Safeexp(max_value=max_value).max_value == max_value


def test_safeexp_max_value_default() -> None:
    assert Safeexp().max_value == 20.0


@mark.parametrize("device", get_available_devices())
def test_safeexp_forward(device: str) -> None:
    device = torch.device(device)
    module = Safeexp().to(device=device)
    assert module(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 485165184.0],
            dtype=torch.float,
            device=device,
        ),
    )


@mark.parametrize("device", get_available_devices())
def test_safeexp_forward_max_value_10(device: str) -> None:
    device = torch.device(device)
    module = Safeexp(max_value=10).to(device=device)
    assert module(torch.tensor([-1, 0, 1, 10, 100], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [0.3678794503211975, 1.0, 2.7182817459106445, 22026.46484375, 22026.46484375],
            dtype=torch.float,
            device=device,
        ),
    )


#############################
#     Tests for Safelog     #
#############################


def test_safelog_str() -> None:
    assert str(Safelog()).startswith("Safelog(")


@mark.parametrize("min_value", (0.1, 0.5))
def test_safelog_min_value(min_value: float) -> None:
    assert Safelog(min_value=min_value).min_value == min_value


def test_safelog_min_value_default() -> None:
    assert Safelog().min_value == 1e-8


@mark.parametrize("device", get_available_devices())
def test_safelog_forward(device: str) -> None:
    device = torch.device(device)
    module = Safelog().to(device=device)
    assert module(torch.tensor([-1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [-18.420680743952367, -18.420680743952367, 0.0, 0.6931471805599453],
            dtype=torch.float,
            device=device,
        ),
    )


@mark.parametrize("device", get_available_devices())
def test_safelog_forward_min_value_1(device: str) -> None:
    device = torch.device(device)
    module = Safelog(min_value=1).to(device=device)
    assert module(torch.tensor([-1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor([0.0, 0.0, 0.0, 0.6931471805599453], dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_safelog_forward_min_value_minus_1(device: str) -> None:
    device = torch.device(device)
    module = Safelog(min_value=-1).to(device=device)
    assert module(torch.tensor([-1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [float("NaN"), float("-inf"), 0.0, 0.6931471805599453], dtype=torch.float, device=device
        ),
        equal_nan=True,
    )


#############################
#     Tests for Squeeze     #
#############################


def test_squeeze_str() -> None:
    assert str(Squeeze()).startswith("Squeeze(")


@mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_none(device: str) -> None:
    device = torch.device(device)
    module = Squeeze().to(device=device)
    assert module(torch.ones(2, 1, 3, 1, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 3, 4, dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_1(device: str) -> None:
    device = torch.device(device)
    module = Squeeze(dim=1).to(device=device)
    assert module(torch.ones(2, 1, 3, 1, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 3, 1, 4, dtype=torch.float, device=device),
    )


@mark.parametrize("device", get_available_devices())
def test_squeeze_forward_dim_2(device: str) -> None:
    device = torch.device(device)
    module = Squeeze(dim=2).to(device=device)
    assert module(torch.ones(2, 1, 3, 1, 4, dtype=torch.float, device=device)).equal(
        torch.ones(2, 1, 3, 1, 4, dtype=torch.float, device=device),
    )


############################
#     Tests for Symlog     #
############################


@mark.parametrize("device", get_available_devices())
def test_symlog_forward(device: str) -> None:
    device = torch.device(device)
    module = Symlog().to(device=device)
    assert module(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [-1.0986122886681098, -0.6931471805599453, 0.0, 0.6931471805599453, 1.0986122886681098],
            dtype=torch.float,
            device=device,
        ),
    )


#############################
#     Tests for Isymlog     #
#############################


@mark.parametrize("device", get_available_devices())
def test_isymlog_forward(device: str) -> None:
    device = torch.device(device)
    module = Isymlog().to(device=device)
    assert module(torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float, device=device)).allclose(
        torch.tensor(
            [-6.38905609893065, -1.718281828459045, 0, 1.718281828459045, 6.38905609893065],
            dtype=torch.float,
            device=device,
        ),
    )


###################################
#     Tests for ToBinaryLabel     #
###################################


def test_to_binary_label_str() -> None:
    assert str(ToBinaryLabel()).startswith("ToBinaryLabel(")


@mark.parametrize("threshold", (0.0, 0.5))
def test_to_binary_label_threshold(threshold: float) -> None:
    assert ToBinaryLabel(threshold=threshold).threshold == threshold


def test_to_binary_label_threshold_default() -> None:
    assert ToBinaryLabel().threshold == 0.0


@mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel().to(device=device)
    assert module(torch.tensor([-1.0, 1.0, -2.0, 1.0], dtype=torch.float, device=device)).equal(
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward_threshold_0_5(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel(threshold=0.5).to(device=device)
    assert module(torch.tensor([0.1, 0.6, 0.4, 1.0], dtype=torch.float, device=device)).equal(
        torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_to_binary_label_forward_2d(device: str) -> None:
    device = torch.device(device)
    module = ToBinaryLabel().to(device=device)
    assert module(
        torch.tensor(
            [[-1.0, 1.0, -2.0, 1.0], [0.0, 1.0, 2.0, -1.0]], dtype=torch.float, device=device
        )
    ).equal(torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long, device=device))


#############################
#     Tests for ToFloat     #
#############################


@mark.parametrize("device", get_available_devices())
def test_to_float_forward(device: str) -> None:
    device = torch.device(device)
    module = ToFloat().to(device=device)
    assert module(torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)).equal(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)
    )


############################
#     Tests for ToLong     #
############################


@mark.parametrize("device", get_available_devices())
def test_to_long_forward(device: str) -> None:
    device = torch.device(device)
    module = ToLong().to(device=device)
    assert module(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)).equal(
        torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    )


########################################
#     Tests for ToCategoricalLabel     #
########################################


@mark.parametrize("device", get_available_devices())
def test_to_categorical_label_forward_1d(device: str) -> None:
    device = torch.device(device)
    module = ToCategoricalLabel().to(device=device)
    assert module(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float, device=device)).equal(
        torch.tensor(3, dtype=torch.long, device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_to_categorical_label_forward_2d(device: str) -> None:
    device = torch.device(device)
    module = ToCategoricalLabel().to(device=device)
    assert module(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]], dtype=torch.float, device=device)
    ).equal(torch.tensor([3, 0], dtype=torch.long, device=device))


#####################################
#     Tests for SequenceToBatch     #
#####################################


@mark.parametrize("device", get_available_devices())
def test_sequence_to_batch_2d(device: str) -> None:
    device = torch.device(device)
    module = SequenceToBatch().to(device=device)
    assert module(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 3.0, 2.0, 2.0]], device=device)).equal(
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 2.0], device=device)
    )


@mark.parametrize("device", get_available_devices())
def test_sequence_to_batch_3d(device: str) -> None:
    device = torch.device(device)
    module = SequenceToBatch().to(device=device)
    assert module(
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 3.0], [2.0, 2.0]]], device=device)
    ).equal(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 3.0], [2.0, 2.0]], device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("dtype", DTYPES)
def test_sequence_to_batch_dtype(device: str, dtype: torch.dtype) -> None:
    device = torch.device(device)
    module = SequenceToBatch().to(device=device)
    assert module(torch.ones(2, 3, dtype=dtype, device=device)).equal(
        torch.ones(6, dtype=dtype, device=device)
    )
