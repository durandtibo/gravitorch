import torch
from pytest import mark, raises
from torch.nn import Linear

from gravitorch.nn import (
    AverageFusion,
    ConcatFusion,
    FusionFFN,
    MultiplicationFusion,
    SumFusion,
)
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


###################################
#     Tests for AverageFusion     #
###################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_1_input(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_2_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(2))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_average_fusion_forward_3_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(2),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(2))


@mark.parametrize("device", get_available_devices())
def test_average_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = AverageFusion().to(device=device)
    y = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    y.mean().backward()
    assert y.equal(torch.ones(2, 4, device=device))


##################################
#     Tests for ConcatFusion     #
##################################


def test_concat_fusion_str() -> None:
    assert str(ConcatFusion()).startswith("ConcatFusion(")


def test_concat_fusion_forward_0_input() -> None:
    module = ConcatFusion()
    with raises(RuntimeError, match="ConcatFusion needs at least one tensor as input"):
        module()


@mark.parametrize("device", get_available_devices())
def test_concat_fusion_forward(device: str) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    assert net(
        torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long, device=device),
        torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.long, device=device),
    ).equal(torch.tensor([[2, 3, 4, 12, 13, 14], [5, 6, 7, 15, 16, 17]], dtype=torch.long, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_1_input(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    assert net(torch.ones(batch_size, 3, device=device)).equal(
        torch.ones(batch_size, 3, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_2_inputs(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    assert net(
        torch.ones(batch_size, 3, device=device), torch.ones(batch_size, 4, device=device)
    ).equal(torch.ones(batch_size, 7, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_3_inputs(device: str, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    assert net(
        torch.ones(batch_size, 3, device=device),
        torch.ones(batch_size, 4, device=device),
        torch.ones(batch_size, 5, device=device),
    ).equal(torch.ones(batch_size, 12, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_forward_3d_inputs(device: str, seq_len: int, batch_size: int) -> None:
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    assert net(
        torch.ones(batch_size, seq_len, 3, device=device),
        torch.ones(batch_size, seq_len, 4, device=device),
    ).equal(torch.ones(batch_size, seq_len, 7, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_concat_fusion_forward_dim_0(
    device: str, seq_len: int, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    net = ConcatFusion(dim=0).to(device=device)
    assert net(
        torch.ones(seq_len, batch_size, input_size, device=device),
        torch.ones(seq_len, batch_size, input_size, device=device),
    ).equal(torch.ones(2 * seq_len, batch_size, input_size, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_concat_fusion_forward_dim_1(
    device: str, seq_len: int, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    net = ConcatFusion(dim=1).to(device=device)
    assert net(
        torch.ones(batch_size, seq_len, input_size, device=device),
        torch.ones(batch_size, seq_len, input_size, device=device),
    ).equal(torch.ones(batch_size, 2 * seq_len, input_size, device=device))


@mark.parametrize("device", get_available_devices())
def test_concat_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = ConcatFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert out.equal(torch.ones(2, 12, device=device))


###############################
#     Tests for FusionFFN     #
###############################


def test_fusion_ffn_modules() -> None:
    module = FusionFFN(fusion=SumFusion(), ffn=Linear(4, 7))
    assert isinstance(module.fusion, SumFusion)
    assert isinstance(module.ffn, Linear)


@mark.parametrize("output_size", SIZES)
def test_fusion_ffn_output_size(output_size: int) -> None:
    assert FusionFFN(fusion=SumFusion(), ffn=Linear(4, output_size)).output_size == output_size


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_fusion_ffn_forward(
    device: str, batch_size: int, input_size: int, output_size: int
) -> None:
    device = torch.device(device)
    net = FusionFFN(fusion=SumFusion(), ffn=Linear(input_size, output_size)).to(device=device)
    out = net(
        torch.ones(batch_size, input_size, device=device),
        3 * torch.ones(batch_size, input_size, device=device),
    )
    assert out.shape == (batch_size, output_size)
    assert out.dtype == torch.float
    assert out.device == device


##########################################
#     Tests for MultiplicationFusion     #
##########################################


def test_multiplication_fusion_forward_0_input() -> None:
    module = MultiplicationFusion()
    with raises(RuntimeError, match="MultiplicationFusion needs at least one tensor as input"):
        module()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_1_input(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_2_inputs(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(0.25))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_multiplication_fusion_forward_3_inputs(
    device: str, batch_size: int, input_size: int
) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
        torch.ones(batch_size, input_size, device=device).mul(0.5),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(0.125))


@mark.parametrize("device", get_available_devices())
def test_multiplication_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = MultiplicationFusion().to(device=device)
    out = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    out.mean().backward()
    assert out.equal(torch.ones(2, 4, device=device))


###############################
#     Tests for SumFusion     #
###############################


def test_sum_fusion_str() -> None:
    assert str(SumFusion()).startswith("SumFusion(")


def test_sum_fusion_forward_0_input() -> None:
    module = SumFusion()
    with raises(RuntimeError, match="SumFusion needs at least one tensor as input"):
        module()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_1_input(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(torch.ones(batch_size, input_size, device=device)).equal(
        torch.ones(batch_size, input_size, device=device)
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_2_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(4))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_sum_fusion_forward_3_inputs(device: str, batch_size: int, input_size: int) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    assert module(
        torch.ones(batch_size, input_size, device=device),
        torch.ones(batch_size, input_size, device=device).mul(2),
        torch.ones(batch_size, input_size, device=device).mul(3),
    ).equal(torch.ones(batch_size, input_size, device=device).mul(6))


@mark.parametrize("device", get_available_devices())
def test_sum_fusion_backward(device: str) -> None:
    device = torch.device(device)
    module = SumFusion().to(device=device)
    y = module(
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
        torch.ones(2, 4, device=device, requires_grad=True),
    )
    y.mean().backward()
    assert y.equal(torch.ones(2, 4, device=device).mul(3))
