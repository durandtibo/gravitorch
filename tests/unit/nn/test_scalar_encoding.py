from collections.abc import Callable
from typing import Union
from unittest.mock import Mock, patch

import torch
from pytest import mark, raises
from torch import Tensor, nn

from gravitorch.nn import (
    AsinhCosSinScalarEncoder,
    AsinhScalarEncoder,
    CosSinScalarEncoder,
    ScalarEncoderFFN,
)
from gravitorch.utils import get_available_devices

SIZES = (1, 2)

########################################
#     Tests for AsinhScalarEncoder     #
########################################

MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
    AsinhScalarEncoder.create_rand_scale,
    AsinhScalarEncoder.create_linspace_scale,
    AsinhScalarEncoder.create_logspace_scale,
)


def test_asinh_scalar_encoder_str() -> None:
    assert str(
        AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=0.1, max_scale=10.0)
    ).startswith("AsinhScalarEncoder(")


@mark.parametrize(
    "scale",
    (
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float),
        [1.0, 2.0, 4.0],
        (1.0, 2.0, 4.0),
    ),
)
def test_asinh_scalar_encoder_scale(scale: Union[Tensor, list[float], tuple[float, ...]]) -> None:
    assert AsinhScalarEncoder(scale).scale.equal(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float))


def test_asinh_scalar_encoder_input_size() -> None:
    assert (
        AsinhScalarEncoder.create_rand_scale(dim=5, min_scale=0.1, max_scale=10.0).input_size == 1
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_2d(
    device: str, batch_size: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("learnable", (True, False))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_backward(
    device: str, batch_size: int, learnable: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(dim=10, min_scale=0.1, max_scale=10.0, learnable=learnable).to(
        device=device
    )
    out = module(torch.rand(batch_size, 1, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("dim", SIZES)
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_dim(dim: int, module_init: Callable) -> None:
    assert module_init(dim=dim, min_scale=0.1, max_scale=10.0).scale.shape == (dim,)


@mark.parametrize("dim", (0, -1))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_dim_incorrect(dim: int, module_init: Callable) -> None:
    with raises(ValueError):
        module_init(dim=dim, min_scale=0.1, max_scale=10.0)


@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_min_scale_incorrect(module_init: Callable) -> None:
    with raises(ValueError):
        module_init(dim=2, min_scale=0, max_scale=1)


@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_max_scale_incorrect(module_init: Callable) -> None:
    with raises(ValueError):
        module_init(dim=2, min_scale=0.1, max_scale=0.01)


@mark.parametrize("learnable", (True, False))
@mark.parametrize("module_init", MODULE_CONSTRUCTORS)
def test_asinh_scalar_encoder_learnable(learnable: bool, module_init: Callable) -> None:
    assert (
        module_init(dim=2, min_scale=0.01, max_scale=1, learnable=learnable).scale.requires_grad
        == learnable
    )


@patch(
    "gravitorch.nn.scalar_encoding.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),
)
def test_asinh_scalar_encoder_create_rand_scale() -> None:
    assert AsinhScalarEncoder.create_rand_scale(dim=3, min_scale=0.2, max_scale=1).scale.data.equal(
        torch.tensor([0.2, 0.6, 1.0])
    )


def test_asinh_scalar_encoder_create_linspace_scale() -> None:
    assert AsinhScalarEncoder.create_linspace_scale(
        dim=3, min_scale=0.2, max_scale=1
    ).scale.data.equal(torch.tensor([0.2, 0.6, 1.0]))


def test_asinh_scalar_encoder_create_logspace_scale() -> None:
    assert AsinhScalarEncoder.create_logspace_scale(
        dim=3, min_scale=0.01, max_scale=1
    ).scale.data.equal(torch.tensor([0.01, 0.1, 1.0]))


def test_asinh_scalar_encoder_forward_scale() -> None:
    module = AsinhScalarEncoder(scale=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float))
    assert module(torch.tensor([[-1], [0], [1]], dtype=torch.float)).allclose(
        torch.tensor(
            [
                [-0.881373587019543, -1.4436354751788103, -1.8184464592320668],
                [0.0, 0.0, 0.0],
                [0.881373587019543, 1.4436354751788103, 1.8184464592320668],
            ],
            dtype=torch.float,
        ),
    )


#########################################
#     Tests for CosSinScalarEncoder     #
#########################################

COSSIN_MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
    CosSinScalarEncoder.create_rand_frequency,
    CosSinScalarEncoder.create_linspace_frequency,
    CosSinScalarEncoder.create_logspace_frequency,
)


def test_cos_sin_scalar_encoder_str() -> None:
    assert str(
        CosSinScalarEncoder.create_rand_frequency(
            num_frequencies=3, min_frequency=0.1, max_frequency=10.0
        )
    ).startswith("CosSinScalarEncoder(")


@mark.parametrize(
    "frequency",
    (
        torch.tensor([1.0, 2.0, 4.0], dtype=torch.float),
        [1.0, 2.0, 4.0],
        (1.0, 2.0, 4.0),
    ),
)
@mark.parametrize(
    "phase_shift",
    (
        torch.tensor([1.0, 3.0, -2.0], dtype=torch.float),
        [1.0, 3.0, -2.0],
        (1.0, 3.0, -2.0),
    ),
)
def test_cos_sin_scalar_encoder_frequency_phase_shift(
    frequency: Union[Tensor, list[float], tuple[float, ...]],
    phase_shift: Union[Tensor, list[float], tuple[float, ...]],
) -> None:
    module = CosSinScalarEncoder(frequency, phase_shift)
    assert module.frequency.equal(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float))
    assert module.phase_shift.equal(torch.tensor([1.0, 3.0, -2.0], dtype=torch.float))


def test_cos_sin_scalar_encoder_frequency_phase_shift_incorrect_dim() -> None:
    with raises(ValueError):
        CosSinScalarEncoder(torch.rand(1, 6), torch.rand(6))


def test_cos_sin_scalar_encoder_frequency_phase_shift_incorrect_shape() -> None:
    with raises(ValueError):
        CosSinScalarEncoder(torch.rand(6), torch.rand(4))


def test_cos_sin_scalar_encoder_input_size() -> None:
    assert (
        CosSinScalarEncoder.create_rand_frequency(
            num_frequencies=3, min_frequency=0.1, max_frequency=10.0
        ).input_size
        == 1
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_2d(
    device: str, batch_size: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("learnable", (True, False))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_backward(
    device: str, batch_size: int, learnable: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(
        num_frequencies=5, min_frequency=0.1, max_frequency=10.0, learnable=learnable
    ).to(device=device)
    out = module(torch.rand(batch_size, 1, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 10)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("dim", SIZES)
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_dim(dim: int, module_init: Callable) -> None:
    module = module_init(num_frequencies=dim, min_frequency=0.1, max_frequency=10.0)
    assert module.frequency.shape == (dim * 2,)
    assert module.phase_shift.shape == (dim * 2,)


@mark.parametrize("dim", (0, -1))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_dim_incorrect(dim: int, module_init: Callable) -> None:
    with raises(ValueError):
        module_init(num_frequencies=dim, min_frequency=0.1, max_frequency=10.0)


@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_min_frequency_incorrect(module_init: Callable) -> None:
    with raises(ValueError):
        module_init(num_frequencies=2, min_frequency=0, max_frequency=1)


@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_max_frequency_incorrect(module_init: Callable) -> None:
    with raises(ValueError):
        module_init(num_frequencies=2, min_frequency=0.1, max_frequency=0.01)


@mark.parametrize("learnable", (True, False))
@mark.parametrize("module_init", COSSIN_MODULE_CONSTRUCTORS)
def test_cos_sin_scalar_encoder_learnable(learnable: bool, module_init: Callable) -> None:
    module = module_init(
        num_frequencies=2, min_frequency=0.01, max_frequency=1, learnable=learnable
    )
    assert module.frequency.requires_grad == learnable
    assert module.phase_shift.requires_grad == learnable


@patch(
    "gravitorch.nn.scalar_encoding.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),
)
def test_cos_sin_scalar_encoder_create_rand_frequency() -> None:
    module = CosSinScalarEncoder.create_rand_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.2, 0.6, 1.0, 0.2, 0.6, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


@patch(
    "gravitorch.nn.scalar_encoding.torch.rand",
    lambda *args, **kwargs: torch.tensor([0.0, 0.5, 1.0]),
)
def test_cos_sin_scalar_encoder_create_rand_value_range() -> None:
    module = CosSinScalarEncoder.create_rand_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 3.0, 5.0, 1.0, 3.0, 5.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_linspace_frequency() -> None:
    module = CosSinScalarEncoder.create_linspace_frequency(
        num_frequencies=3, min_frequency=0.2, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.2, 0.6, 1.0, 0.2, 0.6, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_linspace_value_range() -> None:
    module = CosSinScalarEncoder.create_linspace_value_range(
        num_frequencies=3, min_abs_value=0.2, max_abs_value=1.0
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 3.0, 5.0, 1.0, 3.0, 5.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_logspace_frequency() -> None:
    module = CosSinScalarEncoder.create_logspace_frequency(
        num_frequencies=3, min_frequency=0.01, max_frequency=1
    )
    assert module.frequency.data.equal(torch.tensor([0.01, 0.1, 1.0, 0.01, 0.1, 1.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_create_logspace_value_range() -> None:
    module = CosSinScalarEncoder.create_logspace_value_range(
        num_frequencies=3, min_abs_value=0.01, max_abs_value=1.0
    )
    assert module.frequency.data.equal(torch.tensor([1.0, 10.0, 100.0, 1.0, 10.0, 100.0]))
    assert module.phase_shift.data.equal(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_cos_sin_scalar_encoder_forward_frequency_phase_shift() -> None:
    module = CosSinScalarEncoder(
        frequency=torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float),
        phase_shift=torch.zeros(6),
    )
    assert module(torch.tensor([[-1], [0], [1]], dtype=torch.float)).allclose(
        torch.tensor(
            [
                [
                    -0.8414709848078965,
                    -0.9092974268256817,
                    -0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                ],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [
                    0.8414709848078965,
                    0.9092974268256817,
                    0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                ],
            ],
            dtype=torch.float,
        ),
    )


##############################################
#     Tests for AsinhCosSinScalarEncoder     #
##############################################

ASINH_COSSIN_MODULE_CONSTRUCTORS: tuple[Callable, ...] = (
    AsinhCosSinScalarEncoder.create_rand_frequency,
    AsinhCosSinScalarEncoder.create_linspace_frequency,
    AsinhCosSinScalarEncoder.create_logspace_frequency,
)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", ASINH_COSSIN_MODULE_CONSTRUCTORS)
def test_asinh_cos_sin_scalar_encoder_forward_2d(
    device: str, batch_size: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 11)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", ASINH_COSSIN_MODULE_CONSTRUCTORS)
def test_asinh_cos_sin_scalar_encoder_forward_3d_batch_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 11)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
@mark.parametrize("module_init", ASINH_COSSIN_MODULE_CONSTRUCTORS)
def test_asinh_cos_sin_scalar_encoder_forward_3d_seq_first(
    device: str, batch_size: int, seq_len: int, mode: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(num_frequencies=5, min_frequency=0.1, max_frequency=10.0).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 11)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("learnable", (True, False))
@mark.parametrize("module_init", ASINH_COSSIN_MODULE_CONSTRUCTORS)
def test_asinh_cos_sin_scalar_encoder_backward(
    device: str, batch_size: int, learnable: bool, module_init: Callable
) -> None:
    device = torch.device(device)
    module = module_init(
        num_frequencies=5, min_frequency=0.1, max_frequency=10.0, learnable=learnable
    ).to(device=device)
    out = module(torch.rand(batch_size, 1, device=device, requires_grad=True))
    out.mean().backward()
    assert out.shape == (batch_size, 11)
    assert out.device == device
    assert out.dtype == torch.float


def test_asinh_cos_sin_scalar_encoder_forward_frequency_phase_shift() -> None:
    module = AsinhCosSinScalarEncoder(
        frequency=torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float),
        phase_shift=torch.zeros(6),
    )
    assert module(torch.tensor([[-1], [0], [1]], dtype=torch.float)).allclose(
        torch.tensor(
            [
                [
                    -0.8414709848078965,
                    -0.9092974268256817,
                    -0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                    -0.881373587019543,
                ],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [
                    0.8414709848078965,
                    0.9092974268256817,
                    0.1411200080598672,
                    0.5403023058681398,
                    -0.4161468365471424,
                    -0.9899924966004454,
                    0.881373587019543,
                ],
            ],
            dtype=torch.float,
        ),
    )


######################################
#     Tests for ScalarEncoderFFN     #
######################################


@mark.parametrize("input_size", SIZES)
def test_scalar_encoder_ffn_input_size(input_size: int) -> None:
    assert (
        ScalarEncoderFFN(
            encoder=Mock(spec=nn.Module, input_size=input_size), ffn=Mock(spec=nn.Module)
        ).input_size
        == input_size
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_scalar_encoder_ffn_forward_2d(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    module = ScalarEncoderFFN(
        encoder=CosSinScalarEncoder.create_logspace_frequency(
            num_frequencies=5, min_frequency=0.01, max_frequency=1
        ),
        ffn=nn.Linear(10, 6),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, 1, device=device))
    assert out.shape == (batch_size, 6)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
def test_scalar_encoder_ffn_forward_3d_batch_first(
    device: str,
    batch_size: int,
    seq_len: int,
    mode: bool,
) -> None:
    device = torch.device(device)
    module = ScalarEncoderFFN(
        encoder=CosSinScalarEncoder.create_logspace_frequency(
            num_frequencies=5, min_frequency=0.01, max_frequency=1
        ),
        ffn=nn.Linear(10, 6),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(batch_size, seq_len, 1, device=device))
    assert out.shape == (batch_size, seq_len, 6)
    assert out.device == device
    assert out.dtype == torch.float


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("mode", (True, False))
def test_scalar_encoder_ffn_forward_3d_sequence_first(
    device: str,
    batch_size: int,
    seq_len: int,
    mode: bool,
) -> None:
    device = torch.device(device)
    module = ScalarEncoderFFN(
        encoder=CosSinScalarEncoder.create_logspace_frequency(
            num_frequencies=5, min_frequency=0.01, max_frequency=1
        ),
        ffn=nn.Linear(10, 6),
    ).to(device=device)
    module.train(mode)
    out = module(torch.rand(seq_len, batch_size, 1, device=device))
    assert out.shape == (seq_len, batch_size, 6)
    assert out.device == device
    assert out.dtype == torch.float
