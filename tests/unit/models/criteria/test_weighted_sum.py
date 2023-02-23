from unittest.mock import Mock

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from torch import nn

from gravitorch import constants as ct
from gravitorch.models.criteria import VanillaLoss, WeightedSumLoss

#####################################
#     Tests for WeightedSumLoss     #
#####################################


def test_weighted_sum_loss_str():
    assert str(
        WeightedSumLoss(
            {
                "value": {OBJECT_TARGET: "torch.nn.MSELoss"},
                "time": {OBJECT_TARGET: "torch.nn.L1Loss"},
            }
        )
    ).startswith("WeightedSumLoss(")


def test_weighted_sum_loss_from_dict():
    criterion = WeightedSumLoss(
        {"value": {OBJECT_TARGET: "torch.nn.MSELoss"}, "time": {OBJECT_TARGET: "torch.nn.L1Loss"}}
    )
    assert isinstance(criterion.criteria["value"], nn.MSELoss)
    assert isinstance(criterion.criteria["time"], nn.L1Loss)
    assert criterion._weights == {"value": 1.0, "time": 1.0}


def test_weighted_sum_loss_tensor():
    value_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.8))
    time_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.2))
    criterion = WeightedSumLoss(nn.ModuleDict({"value": value_criterion, "time": time_criterion}))
    prediction = torch.rand(2, 3)
    target = torch.rand(2, 3)
    assert objects_are_equal(
        criterion(prediction, target),
        {
            "loss": torch.tensor(1.0),
            "loss_value": torch.tensor(0.8),
            "loss_time": torch.tensor(0.2),
        },
    )
    assert objects_are_equal(value_criterion.call_args.args, (prediction, target))
    assert objects_are_equal(time_criterion.call_args.args, (prediction, target))


def test_weighted_sum_loss_dict():
    value_criterion = Mock(spec=nn.Module, return_value={ct.LOSS: torch.tensor(0.8)})
    time_criterion = Mock(spec=nn.Module, return_value={ct.LOSS: torch.tensor(0.2)})
    criterion = WeightedSumLoss(nn.ModuleDict({"value": value_criterion, "time": time_criterion}))
    prediction = torch.rand(2, 3)
    target = torch.rand(2, 3)
    assert objects_are_equal(
        criterion(prediction, target),
        {
            "loss": torch.tensor(1.0),
            "loss_value": torch.tensor(0.8),
            "loss_time": torch.tensor(0.2),
        },
    )
    assert objects_are_equal(value_criterion.call_args.args, (prediction, target))
    assert objects_are_equal(time_criterion.call_args.args, (prediction, target))


def test_weighted_sum_loss_one_weight():
    value_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.8))
    time_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.2))
    criterion = WeightedSumLoss(
        nn.ModuleDict(
            {
                "value": value_criterion,
                "time": time_criterion,
            }
        ),
        weights={"value": 0.5},
    )
    prediction = torch.rand(2, 3)
    target = torch.rand(2, 3)
    assert objects_are_equal(
        criterion(prediction, target),
        {
            "loss": torch.tensor(0.6),
            "loss_value": torch.tensor(0.4),
            "loss_time": torch.tensor(0.2),
        },
    )
    assert objects_are_equal(value_criterion.call_args.args, (prediction, target))
    assert objects_are_equal(time_criterion.call_args.args, (prediction, target))


def test_weighted_sum_loss_both_weight():
    value_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.8))
    time_criterion = Mock(spec=nn.Module, return_value=torch.tensor(0.2))
    criterion = WeightedSumLoss(
        nn.ModuleDict(
            {
                "value": value_criterion,
                "time": time_criterion,
            }
        ),
        weights={"value": 0.5, "time": 0.5},
    )
    prediction = torch.rand(2, 3)
    target = torch.rand(2, 3)
    assert objects_are_equal(
        criterion(prediction, target),
        {
            "loss": torch.tensor(0.5),
            "loss_value": torch.tensor(0.4),
            "loss_time": torch.tensor(0.1),
        },
    )
    assert objects_are_equal(value_criterion.call_args.args, (prediction, target))
    assert objects_are_equal(time_criterion.call_args.args, (prediction, target))


def test_weighted_sum_loss_pytorch_losses():
    criterion = WeightedSumLoss(nn.ModuleDict({"value": nn.MSELoss(), "time": nn.L1Loss()}))
    target = torch.rand(2, 3)
    out = criterion(target, target)
    assert out[ct.LOSS].equal(torch.tensor(0.0))
    assert out[f"{ct.LOSS}_time"].equal(torch.tensor(0.0))
    assert out[f"{ct.LOSS}_value"].equal(torch.tensor(0.0))


def test_weighted_sum_loss_gravitorch_losses():
    criterion = WeightedSumLoss(
        nn.ModuleDict({"value": VanillaLoss(nn.MSELoss()), "time": VanillaLoss(nn.L1Loss())})
    )
    target = torch.rand(2, 3)
    out = criterion({ct.PREDICTION: target}, {ct.TARGET: target})
    assert out[ct.LOSS].equal(torch.tensor(0.0))
    assert out[f"{ct.LOSS}_time"].equal(torch.tensor(0.0))
    assert out[f"{ct.LOSS}_value"].equal(torch.tensor(0.0))
