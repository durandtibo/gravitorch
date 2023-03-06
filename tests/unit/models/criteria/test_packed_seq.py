from typing import Union

import torch
from coola import objects_are_allclose, objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark
from torch.nn import CrossEntropyLoss, Module, MSELoss
from torch.nn.utils.rnn import pack_sequence

from gravitorch import constants as ct
from gravitorch.models.criteria import PackedSequenceLoss
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


########################################
#     Tests for PackedSequenceLoss     #
########################################


@mark.parametrize(
    "criterion,criterion_cls",
    (
        (MSELoss(), MSELoss),
        (CrossEntropyLoss(), CrossEntropyLoss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, MSELoss),
    ),
)
def test_packed_sequence_loss_criterion(
    criterion: Union[dict, Module], criterion_cls: type[Module]
):
    assert isinstance(PackedSequenceLoss(criterion).criterion, criterion_cls)


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_mse_correct(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: pack_sequence([torch.ones(3, 4, device=device) for _ in range(2)])},
            {ct.TARGET: pack_sequence([torch.ones(3, 4, device=device) for _ in range(2)])},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_mse_incorrect_invalid_1(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1], [2], [3]], dtype=torch.float, device=device),
                        torch.tensor([[4], [5]], dtype=torch.float, device=device),
                    ]
                )
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([[2], [1], [2]], dtype=torch.float, device=device),
                        torch.tensor([[4], [3]], dtype=torch.float, device=device),
                    ]
                )
            },
        ),
        {ct.LOSS: torch.tensor(1.4, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_mse_incorrect_invalid_2(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1], [2], [3]], dtype=torch.float, device=device),
                        torch.tensor([[4]], dtype=torch.float, device=device),
                    ]
                )
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([[2], [1], [2]], dtype=torch.float, device=device),
                        torch.tensor([[4]], dtype=torch.float, device=device),
                    ]
                )
            },
        ),
        {ct.LOSS: torch.tensor(0.75, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_mse_batch_size_1(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [torch.tensor([[1], [2]], dtype=torch.float, device=device)]
                )
            },
            {
                ct.TARGET: pack_sequence(
                    [torch.tensor([[2], [1]], dtype=torch.float, device=device)]
                )
            },
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_mse_with_mask(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1], [2], [3]], dtype=torch.float, device=device),
                        torch.tensor([[4], [5], [0]], dtype=torch.float, device=device),
                    ]
                ),
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([[2], [1], [2]], dtype=torch.float, device=device),
                        torch.tensor([[4], [3], [6]], dtype=torch.float, device=device),
                    ]
                ),
                ct.MASK: pack_sequence(
                    [torch.tensor([1, 1, 1], device=device), torch.tensor([1, 1, 0], device=device)]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.4, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_cross_entropy(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: pack_sequence([torch.ones(3, 4, device=device) for _ in range(2)]),
            },
            {
                ct.TARGET: pack_sequence(
                    [torch.ones(3, dtype=torch.long, device=device) for _ in range(2)]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.3862943611198906, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_cross_entropy_all_valid(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.float, device=device),
                        torch.tensor([[2, 1, 3], [1, 2, 3]], dtype=torch.float, device=device),
                    ]
                ),
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                    ]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.1576059644443804, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_cross_entropy_invalid_1(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.float, device=device),
                        torch.tensor([[2, 1, 3]], dtype=torch.float, device=device),
                    ]
                ),
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                        torch.tensor([1], dtype=torch.long, device=device),
                    ]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.4076059644443804, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_cross_entropy_batch_size_1(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.float, device=device)]
                ),
            },
            {
                ct.TARGET: pack_sequence([torch.tensor([1, 2], dtype=torch.long, device=device)]),
            },
        ),
        {ct.LOSS: torch.tensor(0.9076059644443802, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_packed_sequence_cross_entropy_with_mask(device: str) -> None:
    device = torch.device(device)
    criterion = PackedSequenceLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                ct.PREDICTION: pack_sequence(
                    [
                        torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.float, device=device),
                        torch.tensor([[2, 1, 3], [1, 2, 3]], dtype=torch.float, device=device),
                    ]
                ),
            },
            {
                ct.TARGET: pack_sequence(
                    [
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                    ]
                ),
                ct.MASK: pack_sequence(
                    [torch.tensor([1, 1], device=device), torch.tensor([1, 0], device=device)]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.4076059644443804, device=device)},
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("prediction_key", ("my_prediction", "output"))
@mark.parametrize("target_key", ("my_target", "target"))
@mark.parametrize("mask_key", ("my_mask", "mask"))
def test_packed_sequence_cross_entropy_custom_keys(
    device: str, prediction_key: str, target_key: str, mask_key: str
):
    device = torch.device(device)
    criterion = PackedSequenceLoss(
        CrossEntropyLoss(),
        prediction_key=prediction_key,
        target_key=target_key,
        mask_key=mask_key,
    ).to(device=device)
    assert objects_are_allclose(
        criterion(
            {
                prediction_key: pack_sequence(
                    [
                        torch.tensor([[1, 2, 3], [2, 1, 3]], dtype=torch.float, device=device),
                        torch.tensor([[2, 1, 3], [1, 2, 3]], dtype=torch.float, device=device),
                    ]
                ),
            },
            {
                target_key: pack_sequence(
                    [
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                        torch.tensor([1, 2], dtype=torch.long, device=device),
                    ]
                ),
                mask_key: pack_sequence(
                    [torch.tensor([1, 1], device=device), torch.tensor([1, 0], device=device)]
                ),
            },
        ),
        {ct.LOSS: torch.tensor(1.4076059644443804, device=device)},
    )
