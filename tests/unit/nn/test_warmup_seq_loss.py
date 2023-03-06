from typing import Union

import torch
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch.nn import WarmupSequenceLoss
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


########################################
#     Tests for WarmupSequenceLoss     #
########################################


@mark.parametrize(
    "criterion,criterion_cls",
    (
        (nn.MSELoss(), nn.MSELoss),
        (nn.CrossEntropyLoss(), nn.CrossEntropyLoss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, nn.MSELoss),
    ),
)
def test_warmup_sequence_loss_criterion(
    criterion: Union[dict, nn.Module], criterion_cls: type[nn.Module]
) -> None:
    assert isinstance(WarmupSequenceLoss(criterion=criterion).criterion, criterion_cls)


@mark.parametrize("warmup", (0, 1, 2))
def test_warmup_sequence_loss_warmup(warmup: int) -> None:
    assert WarmupSequenceLoss(criterion=nn.MSELoss(), warmup=warmup)._warmup == warmup


@mark.parametrize("batch_first", (True, False))
def test_warmup_sequence_loss_batch_first(batch_first: bool) -> None:
    assert (
        WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=batch_first)._batch_first
        == batch_first
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_first", (True, False))
@mark.parametrize("warmup", (0, 1, 2))
def test_warmup_sequence_mse_2d(device: str, batch_first: bool, warmup: int) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.MSELoss(), batch_first=batch_first, warmup=warmup
    ).to(device=device)
    loss = criterion(torch.ones(6, 5, device=device), torch.ones(6, 5, device=device))
    assert loss.equal(torch.tensor(0.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_2d_batch_first_warmup_1(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True, warmup=1).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float, device=device),
        torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(3.75, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_2d_batch_first_warmup_2(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True, warmup=2).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float, device=device),
        torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(5, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_first", (True, False))
@mark.parametrize("warmup", (0, 1, 2))
def test_warmup_sequence_mse_3d(device: str, batch_first: bool, warmup: int) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.MSELoss(), batch_first=batch_first, warmup=warmup
    ).to(device=device)
    loss = criterion(torch.ones(6, 5, 4, device=device), torch.ones(6, 5, 4, device=device))
    assert loss.equal(torch.tensor(0.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_warmup_sequence_mse_batch_first_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True).to(device=device)
    loss = criterion(
        torch.ones(batch_size, seq_len, feature_size, device=device),
        torch.ones(batch_size, seq_len, feature_size, device=device),
    )
    assert loss.equal(torch.tensor(0.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_3d_batch_first_warmup_1(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True, warmup=1).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[[1], [2], [3]], [[1], [1], [1]]], dtype=torch.float, device=device),
        torch.tensor([[[0], [0], [0]], [[0], [0], [0]]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(3.75, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_3d_batch_first_warmup_2(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True, warmup=2).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[[1], [2], [3]], [[1], [1], [1]]], dtype=torch.float, device=device),
        torch.tensor([[[0], [0], [0]], [[0], [0], [0]]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(5, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_warmup_sequence_mse_sequence_first_3d(
    device: str, batch_size: int, seq_len: int, feature_size: int
) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=True).to(device=device)
    loss = criterion(
        torch.ones(seq_len, batch_size, feature_size, device=device),
        torch.ones(seq_len, batch_size, feature_size, device=device),
    )
    assert loss.equal(torch.tensor(0.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_3d_sequence_first_warmup_1(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=False, warmup=1).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[[1], [1]], [[2], [1]], [[3], [1]]], dtype=torch.float, device=device),
        torch.tensor([[[0], [0]], [[0], [0]], [[0], [0]]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(3.75, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_mse_3d_sequence_first_warmup_2(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.MSELoss(), batch_first=False, warmup=2).to(
        device=device
    )
    loss = criterion(
        torch.tensor([[[1], [1]], [[2], [1]], [[3], [1]]], dtype=torch.float, device=device),
        torch.tensor([[[0], [0]], [[0], [0]], [[0], [0]]], dtype=torch.float, device=device),
    )
    assert loss.equal(torch.tensor(5, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_first", (True, False))
@mark.parametrize("warmup", (0, 1, 2))
def test_warmup_sequence_mse_4d(device: str, batch_first: bool, warmup: int) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.MSELoss(), batch_first=batch_first, warmup=warmup
    ).to(device=device)
    loss = criterion(torch.ones(6, 5, 4, 3, device=device), torch.ones(6, 5, 4, 3, device=device))
    assert loss.equal(torch.tensor(0.0, dtype=torch.float, device=device))


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_first", (True, False))
@mark.parametrize("warmup", (0, 1, 2))
def test_warmup_sequence_cross_entropy(device: str, batch_first: bool, warmup: int) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.CrossEntropyLoss(), batch_first=batch_first, warmup=warmup
    ).to(device=device)
    loss = criterion(
        torch.ones(6, 5, 4, device=device), torch.ones(6, 5, device=device, dtype=torch.long)
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert loss.dtype == torch.float


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_cross_entropy_batch_first_warmup_1(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.CrossEntropyLoss(), batch_first=True, warmup=1).to(
        device=device
    )
    loss = criterion(
        torch.tensor(
            [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]], dtype=torch.float, device=device
        ),
        torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
    )
    assert torch.isclose(loss, torch.tensor(0.40760597586631775, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_cross_entropy_batch_first_warmup_1_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.CrossEntropyLoss(reduction="none"), batch_first=True, warmup=1
    ).to(device=device)
    loss = criterion(
        torch.tensor(
            [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]], dtype=torch.float, device=device
        ),
        torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
    )
    assert torch.all(
        torch.isclose(loss, torch.tensor([0.40760597586631775, 0.40760597586631775], device=device))
    )


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_cross_entropy_sequence_first_warmup_1(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(criterion=nn.CrossEntropyLoss(), batch_first=False, warmup=1).to(
        device=device
    )
    loss = criterion(
        torch.tensor(
            [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]], dtype=torch.float, device=device
        ),
        torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
    )
    assert torch.isclose(loss, torch.tensor(1.4076058864593506, device=device))


@mark.parametrize("device", get_available_devices())
def test_warmup_sequence_cross_entropy_sequence_first_warmup_1_reduction_none(device: str) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.CrossEntropyLoss(reduction="none"), batch_first=False, warmup=1
    ).to(device=device)
    loss = criterion(
        torch.tensor(
            [[[1, 2, 3], [2, 1, 3]], [[2, 1, 3], [1, 2, 3]]], dtype=torch.float, device=device
        ),
        torch.tensor([[1, 2], [1, 2]], dtype=torch.long, device=device),
    )
    assert torch.all(
        torch.isclose(loss, torch.tensor([2.4076058864593506, 0.40760597586631775], device=device))
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_warmup_sequence_cross_entropy_batch_first_reduction_none(
    device: str,
    batch_size: int,
    seq_len: int,
    feature_size: int,
) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.CrossEntropyLoss(reduction="none"), batch_first=True
    ).to(device=device)
    loss = criterion(
        torch.ones(batch_size, seq_len, 3, feature_size, dtype=torch.float, device=device),
        torch.ones(batch_size, seq_len, feature_size, dtype=torch.long, device=device),
    )
    assert loss.shape == (seq_len * batch_size, feature_size)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("feature_size", SIZES)
def test_warmup_sequence_cross_entropy_sequence_first_reduction_none(
    device: str,
    batch_size: int,
    seq_len: int,
    feature_size: int,
) -> None:
    device = torch.device(device)
    criterion = WarmupSequenceLoss(
        criterion=nn.CrossEntropyLoss(reduction="none"), batch_first=False
    ).to(device=device)
    loss = criterion(
        torch.ones(seq_len, batch_size, 3, feature_size, dtype=torch.float, device=device),
        torch.ones(seq_len, batch_size, feature_size, dtype=torch.long, device=device),
    )
    assert loss.shape == (seq_len * batch_size, feature_size)
