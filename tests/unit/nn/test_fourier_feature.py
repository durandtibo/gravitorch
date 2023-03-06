from unittest.mock import patch

import torch
from pytest import mark

from gravitorch.nn import ScaleShiftSequenceGaussianRFF, SequenceGaussianRFF
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#########################################
#     Tests for SequenceGaussianRFF     #
#########################################


def test_sequence_gaussian_rff_str() -> None:
    assert str(SequenceGaussianRFF(input_size=2, output_size=4)).startswith("SequenceGaussianRFF(")


@mark.parametrize("batch_first", (True, False))
def test_sequence_gaussian_rff_batch_first(batch_first: bool) -> None:
    assert (
        SequenceGaussianRFF(input_size=2, output_size=4, batch_first=batch_first).batch_first
        == batch_first
    )


@mark.parametrize("input_size", SIZES)
def test_sequence_gaussian_rff_input_size(input_size: int) -> None:
    assert SequenceGaussianRFF(input_size=input_size, output_size=4).input_size == input_size


@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_sequence_gaussian_rff_gaussian_shape(input_size: int, output_size: int) -> None:
    assert SequenceGaussianRFF(
        input_size=input_size, output_size=output_size, batch_first=True
    ).gaussian.data.shape == (input_size, output_size // 2)


@mark.parametrize("sigma", SIZES)
@patch("gravitorch.nn.fourier_feature.torch.randn", lambda *args, **kwargs: torch.ones(2, 6))
def test_sequence_gaussian_rff_sigma(sigma: float) -> None:
    module = SequenceGaussianRFF(input_size=2, output_size=6, sigma=sigma)
    assert module._sigma == sigma
    assert module.gaussian.equal(sigma * torch.ones(2, 6))


def test_sequence_gaussian_rff_trainable_params_false() -> None:
    assert not SequenceGaussianRFF(input_size=2, output_size=6).gaussian.requires_grad


def test_sequence_gaussian_rff_trainable_params_true() -> None:
    assert SequenceGaussianRFF(
        input_size=2, output_size=6, trainable_params=True
    ).gaussian.requires_grad


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sequence_gaussian_rff_get_dummy_input_batch_first(
    device: str, batch_size: int, seq_len: int, input_size: int
) -> None:
    device = torch.device(device)
    module = SequenceGaussianRFF(input_size=input_size, output_size=6, batch_first=True).to(
        device=device
    )
    dummy_inputs = module.get_dummy_input(batch_size=batch_size, seq_len=seq_len)
    assert len(dummy_inputs) == 1
    assert dummy_inputs[0].shape == (batch_size, seq_len, input_size)
    assert dummy_inputs[0].dtype == torch.float
    assert dummy_inputs[0].device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sequence_gaussian_rff_get_dummy_input_sequence_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
) -> None:
    device = torch.device(device)
    module = SequenceGaussianRFF(input_size=input_size, output_size=6).to(device=device)
    dummy_inputs = module.get_dummy_input(batch_size=batch_size, seq_len=seq_len)
    assert len(dummy_inputs) == 1
    assert dummy_inputs[0].shape == (seq_len, batch_size, input_size)
    assert dummy_inputs[0].dtype == torch.float
    assert dummy_inputs[0].device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_sequence_gaussian_rff_forward_batch_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
    output_size: int,
) -> None:
    device = torch.device(device)
    module = SequenceGaussianRFF(
        input_size=input_size, output_size=output_size, batch_first=True
    ).to(device=device)
    out = module(torch.rand(batch_size, seq_len, input_size, device=device))
    assert out.shape == (batch_size, seq_len, output_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_sequence_gaussian_rff_forward_sequence_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
    output_size: int,
) -> None:
    device = torch.device(device)
    module = SequenceGaussianRFF(input_size=input_size, output_size=output_size).to(device=device)
    out = module(torch.rand(seq_len, batch_size, input_size, device=device))
    assert out.shape == (seq_len, batch_size, output_size)
    assert out.dtype == torch.float
    assert out.device == device


###################################################
#     Tests for ScaleShiftSequenceGaussianRFF     #
###################################################


@mark.parametrize("batch_first", (True, False))
def test_scale_shift_sequence_gaussian_rff_batch_first(batch_first: bool) -> None:
    module = ScaleShiftSequenceGaussianRFF(input_size=2, output_size=4, batch_first=batch_first)
    assert module.batch_first == batch_first
    assert module.shift_scale.batch_first == batch_first
    assert module.rff.batch_first == batch_first


@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_scale_shift_sequence_gaussian_rff_gaussian_shape(
    input_size: int, output_size: int
) -> None:
    module = ScaleShiftSequenceGaussianRFF(
        input_size=input_size, output_size=output_size, batch_first=True
    )
    assert module.rff.gaussian.data.shape == (input_size, output_size // 2)


@mark.parametrize("sigma", SIZES)
@patch("gravitorch.nn.fourier_feature.torch.randn", lambda *args, **kwargs: torch.ones(2, 6))
def test_scale_shift_sequence_gaussian_rff_sigma(sigma: float) -> None:
    module = ScaleShiftSequenceGaussianRFF(input_size=2, output_size=6, sigma=sigma)
    assert module.rff.gaussian.equal(sigma * torch.ones(2, 6))


def test_scale_shift_sequence_gaussian_rff_trainable_params_false() -> None:
    assert not ScaleShiftSequenceGaussianRFF(input_size=2, output_size=6).rff.gaussian.requires_grad


def test_scale_shift_sequence_gaussian_rff_trainable_params_true() -> None:
    assert ScaleShiftSequenceGaussianRFF(
        input_size=2, output_size=6, trainable_params=True
    ).rff.gaussian.requires_grad


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
def test_scale_shift_sequence_gaussian_rff_get_dummy_input_batch_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
) -> None:
    device = torch.device(device)
    module = ScaleShiftSequenceGaussianRFF(
        input_size=input_size, output_size=6, batch_first=True
    ).to(device=device)
    dummy_inputs = module.get_dummy_input(batch_size=batch_size, seq_len=seq_len)
    assert len(dummy_inputs) == 3
    assert dummy_inputs[0].shape == (batch_size, seq_len, input_size)
    assert dummy_inputs[0].dtype == torch.float
    assert dummy_inputs[0].device == device
    assert dummy_inputs[1].shape == (batch_size, 2, input_size)
    assert dummy_inputs[1].dtype == torch.float
    assert dummy_inputs[1].device == device
    assert dummy_inputs[2].shape == (batch_size, 2, input_size)
    assert dummy_inputs[2].dtype == torch.float
    assert dummy_inputs[2].device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
def test_scale_shift_sequence_gaussian_rff_get_dummy_input_sequence_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
) -> None:
    device = torch.device(device)
    module = ScaleShiftSequenceGaussianRFF(input_size=input_size, output_size=6).to(device=device)
    dummy_inputs = module.get_dummy_input(batch_size=batch_size, seq_len=seq_len)
    assert len(dummy_inputs) == 3
    assert dummy_inputs[0].shape == (seq_len, batch_size, input_size)
    assert dummy_inputs[0].dtype == torch.float
    assert dummy_inputs[0].device == device
    assert dummy_inputs[1].shape == (batch_size, 2, input_size)
    assert dummy_inputs[1].dtype == torch.float
    assert dummy_inputs[1].device == device
    assert dummy_inputs[2].shape == (batch_size, 2, input_size)
    assert dummy_inputs[2].dtype == torch.float
    assert dummy_inputs[2].device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_scale_shift_sequence_gaussian_rff_forward_batch_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
    output_size: int,
) -> None:
    device = torch.device(device)
    src_range = torch.ones(batch_size, 2, input_size, device=device)
    src_range[:, 0] = 0
    dst_range = torch.ones(batch_size, 2, input_size, device=device)
    dst_range[:, 0] = 0
    module = ScaleShiftSequenceGaussianRFF(
        input_size=input_size, output_size=output_size, batch_first=True
    ).to(device=device)
    out = module(torch.rand(batch_size, seq_len, input_size, device=device), src_range, dst_range)
    assert out.shape == (batch_size, seq_len, output_size)
    assert out.dtype == torch.float
    assert out.device == device


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", (2, 4))
def test_scale_shift_sequence_gaussian_rff_forward_sequence_first(
    device: str,
    batch_size: int,
    seq_len: int,
    input_size: int,
    output_size: int,
) -> None:
    device = torch.device(device)
    src_range = torch.ones(batch_size, 2, input_size, device=device)
    src_range[:, 0] = 0
    dst_range = torch.ones(batch_size, 2, input_size, device=device)
    dst_range[:, 0] = 0
    module = ScaleShiftSequenceGaussianRFF(input_size=input_size, output_size=output_size).to(
        device=device
    )
    out = module(torch.rand(seq_len, batch_size, input_size, device=device), src_range, dst_range)
    assert out.shape == (seq_len, batch_size, output_size)
    assert out.dtype == torch.float
    assert out.device == device
