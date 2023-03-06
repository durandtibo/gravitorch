import torch
from pytest import mark

from gravitorch.nn import ConcatFusion
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


##################################
#     Tests for ConcatFusion     #
##################################


def test_concat_fusion_str() -> None:
    assert str(ConcatFusion()).startswith("ConcatFusion(")


def test_concat_fusion() -> None:
    x1 = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
    x2 = torch.tensor([[12, 13, 14], [15, 16, 17]], dtype=torch.long)
    net = ConcatFusion()
    y = net(x1, x2)
    assert y.equal(torch.tensor([[2, 3, 4, 12, 13, 14], [5, 6, 7, 15, 16, 17]], dtype=torch.long))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_1_input(device: str, batch_size: int):
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    y = net(torch.ones(batch_size, 3, device=device))
    assert y.equal(torch.ones(batch_size, 3, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_2_inputs(device: str, batch_size: int):
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    y = net(torch.ones(batch_size, 3, device=device), torch.ones(batch_size, 4, device=device))
    assert y.equal(torch.ones(batch_size, 7, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_3_inputs(device: str, batch_size: int):
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    y = net(
        torch.ones(batch_size, 3, device=device),
        torch.ones(batch_size, 4, device=device),
        torch.ones(batch_size, 5, device=device),
    )
    assert y.equal(torch.ones(batch_size, 12, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
def test_concat_fusion_3d_inputs(device: str, seq_len: int, batch_size: int):
    device = torch.device(device)
    net = ConcatFusion().to(device=device)
    y = net(
        torch.ones(batch_size, seq_len, 3, device=device),
        torch.ones(batch_size, seq_len, 4, device=device),
    )
    assert y.equal(torch.ones(batch_size, seq_len, 7, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_concat_fusion_fusion_dim_0(device: str, seq_len: int, batch_size: int, input_size: int):
    device = torch.device(device)
    net = ConcatFusion(dim=0).to(device=device)
    y = net(
        torch.ones(seq_len, batch_size, input_size, device=device),
        torch.ones(seq_len, batch_size, input_size, device=device),
    )
    assert y.equal(torch.ones(2 * seq_len, batch_size, input_size, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_concat_fusion_fusion_dim_1(device: str, seq_len: int, batch_size: int, input_size: int):
    device = torch.device(device)
    net = ConcatFusion(dim=1).to(device=device)
    y = net(
        torch.ones(batch_size, seq_len, input_size, device=device),
        torch.ones(batch_size, seq_len, input_size, device=device),
    )
    assert y.equal(torch.ones(batch_size, 2 * seq_len, input_size, device=device))
