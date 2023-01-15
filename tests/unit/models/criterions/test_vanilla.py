from typing import Union

import torch
from coola import objects_are_allclose, objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark
from torch.nn import CrossEntropyLoss, L1Loss, Module, MSELoss

from gravitorch import constants as ct
from gravitorch.models.criterions import VanillaLoss
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


#################################
#     Tests for VanillaLoss     #
#################################


@mark.parametrize(
    "criterion,criterion_cls",
    (
        (MSELoss(), MSELoss),
        (CrossEntropyLoss(), CrossEntropyLoss),
        ({OBJECT_TARGET: "torch.nn.MSELoss"}, MSELoss),
    ),
)
def test_vanilla_sequence_loss_criterion(
    criterion: Union[dict, Module], criterion_cls: type[Module]
):
    assert isinstance(VanillaLoss(criterion).criterion, criterion_cls)


@mark.parametrize("device", get_available_devices())
def test_vanilla_loss_mse_correct(device: str):
    device = torch.device(device)
    criterion = VanillaLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_vanilla_loss_mse_incorrect(device: str):
    device = torch.device(device)
    criterion = VanillaLoss(MSELoss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_vanilla_loss_l1_correct(device: str):
    device = torch.device(device)
    criterion = VanillaLoss(L1Loss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_vanilla_loss_l1_incorrect(device: str):
    device = torch.device(device)
    criterion = VanillaLoss(L1Loss()).to(device=device)
    assert objects_are_equal(
        criterion(
            {ct.PREDICTION: torch.zeros(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0, device=device)},
    )


@mark.parametrize("device", get_available_devices())
def test_vanilla_loss_cross_entropy(device: str):
    device = torch.device(device)
    criterion = VanillaLoss(CrossEntropyLoss()).to(device=device)
    assert objects_are_allclose(
        criterion(
            {ct.PREDICTION: torch.ones(2, 3, device=device)},
            {ct.TARGET: torch.ones(2, dtype=torch.long, device=device)},
        ),
        {ct.LOSS: torch.tensor(1.0986122886681098, device=device)},
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("prediction_key", ("my_prediction", "output"))
@mark.parametrize("target_key", ("my_target", "target"))
def test_vanilla_loss_mse_custom_keys(device: str, prediction_key: str, target_key: str):
    device = torch.device(device)
    criterion = VanillaLoss(MSELoss(), prediction_key=prediction_key, target_key=target_key).to(
        device=device
    )
    assert objects_are_equal(
        criterion(
            {prediction_key: torch.ones(2, 3, device=device)},
            {target_key: torch.ones(2, 3, device=device)},
        ),
        {ct.LOSS: torch.tensor(0.0, device=device)},
    )
