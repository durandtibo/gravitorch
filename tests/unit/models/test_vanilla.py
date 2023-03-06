from pathlib import Path
from unittest.mock import Mock, patch

import torch
from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from gravitorch import constants as ct
from gravitorch.engines.events import EngineEvents
from gravitorch.models.criteria import VanillaLoss
from gravitorch.models.metrics import CategoricalAccuracy, VanillaMetric
from gravitorch.models.networks import BetaMLP
from gravitorch.models.vanilla import VanillaModel
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler

SIZES = (1, 2)


##################################
#     Tests for VanillaModel     #
##################################

# TODO: add GRU based model


def test_vanilla_model_init_mlp_without_metric() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert isinstance(model.metrics, nn.ModuleDict)
    assert len(model.metrics) == 0


def test_vanilla_model_init_mlp_with_metrics() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        metrics=nn.ModuleDict(
            {
                f"{ct.TRAIN}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.TRAIN)),
                f"{ct.EVAL}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.EVAL)),
            }
        ),
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert isinstance(model.metrics, nn.ModuleDict)
    assert isinstance(model.metrics[f"{ct.TRAIN}_metric"], VanillaMetric)
    assert isinstance(model.metrics[f"{ct.EVAL}_metric"], VanillaMetric)
    assert len(model.metrics) == 2


def test_vanilla_model_init_mlp_without_metric_from_config() -> None:
    model = VanillaModel(
        network={
            OBJECT_TARGET: "gravitorch.models.networks.BetaMLP",
            "input_size": 16,
            "hidden_sizes": (32, 8),
        },
        criterion={
            OBJECT_TARGET: "gravitorch.models.criteria.VanillaLoss",
            "criterion": {OBJECT_TARGET: "torch.nn.CrossEntropyLoss"},
        },
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert len(model.metrics) == 0


def test_vanilla_model_init_mlp_with_train_metric_from_config() -> None:
    model = VanillaModel(
        network={
            OBJECT_TARGET: "gravitorch.models.networks.BetaMLP",
            "input_size": 16,
            "hidden_sizes": (32, 8),
        },
        criterion={
            OBJECT_TARGET: "gravitorch.models.criteria.VanillaLoss",
            "criterion": "torch.nn.CrossEntropyLoss",
        },
        metrics={
            f"{ct.EVAL}_metric": {
                OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                "metric": {
                    OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                    "mode": ct.EVAL,
                },
            },
        },
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert isinstance(model.metrics, nn.ModuleDict)
    assert isinstance(model.metrics[f"{ct.EVAL}_metric"], VanillaMetric)
    assert len(model.metrics) == 1


def test_vanilla_model_init_mlp_with_metrics_from_config() -> None:
    model = VanillaModel(
        network={
            OBJECT_TARGET: "gravitorch.models.networks.BetaMLP",
            "input_size": 16,
            "hidden_sizes": (32, 8),
        },
        criterion={
            OBJECT_TARGET: "gravitorch.models.criteria.VanillaLoss",
            "criterion": "torch.nn.CrossEntropyLoss",
        },
        metrics={
            f"{ct.TRAIN}_metric": {
                OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                "metric": {
                    OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                    "mode": ct.TRAIN,
                },
            },
            f"{ct.EVAL}_metric": {
                OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                "metric": {
                    OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                    "mode": ct.EVAL,
                },
            },
        },
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert isinstance(model.metrics, nn.ModuleDict)
    assert isinstance(model.metrics[f"{ct.TRAIN}_metric"], VanillaMetric)
    assert isinstance(model.metrics[f"{ct.EVAL}_metric"], VanillaMetric)
    assert len(model.metrics) == 2


def test_vanilla_model_init_mlp_with_hybrid_metrics() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        metrics={
            f"{ct.TRAIN}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.TRAIN)),
            f"{ct.EVAL}_metric": {
                OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                "metric": {
                    OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                    "mode": ct.EVAL,
                },
            },
        },
    )
    assert isinstance(model.network, BetaMLP)
    assert isinstance(model.criterion, VanillaLoss)
    assert isinstance(model.metrics, nn.ModuleDict)
    assert isinstance(model.metrics[f"{ct.TRAIN}_metric"], VanillaMetric)
    assert isinstance(model.metrics[f"{ct.EVAL}_metric"], VanillaMetric)
    assert len(model.metrics) == 2


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("mode", (True, False))
def test_vanilla_model_forward_mlp_without_metric(device: str, batch_size: int, mode: bool) -> None:
    device = torch.device(device)
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
    ).to(device=device)
    model.train(mode)
    output = model(
        {
            ct.INPUT: torch.ones(batch_size, 16, device=device),
            ct.TARGET: torch.zeros(batch_size, dtype=torch.long, device=device),
        }
    )
    assert len(output) == 2
    assert torch.is_tensor(output[ct.LOSS])
    assert output[ct.PREDICTION].size() == (batch_size, 8)


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_vanilla_model_forward_mlp_with_metric_train(device: str, batch_size: int):
    device = torch.device(device)
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        metrics=nn.ModuleDict(
            {
                f"{ct.TRAIN}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.TRAIN)),
                f"{ct.EVAL}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.EVAL)),
            }
        ),
    ).to(device=device)
    output = model(
        {
            ct.INPUT: torch.ones(batch_size, 16, device=device),
            ct.TARGET: torch.zeros(batch_size, dtype=torch.long, device=device),
        }
    )
    assert len(output) == 2
    assert torch.is_tensor(output[ct.LOSS])
    assert output[ct.PREDICTION].size() == (batch_size, 8)
    assert model.metrics[f"{ct.TRAIN}_metric"].metric._state.num_predictions == batch_size
    assert model.metrics[f"{ct.EVAL}_metric"].metric._state.num_predictions == 0


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_vanilla_model_forward_mlp_with_metric_eval(device: str, batch_size: int):
    device = torch.device(device)
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        metrics=nn.ModuleDict(
            {
                f"{ct.TRAIN}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.TRAIN)),
                f"{ct.EVAL}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.EVAL)),
            }
        ),
    ).to(device=device)
    model.eval()
    output = model(
        {
            ct.INPUT: torch.ones(batch_size, 16, device=device),
            ct.TARGET: torch.zeros(batch_size, dtype=torch.long, device=device),
        }
    )
    assert len(output) == 2
    assert torch.is_tensor(output[ct.LOSS])
    assert output[ct.PREDICTION].size() == (batch_size, 8)
    assert model.metrics[f"{ct.TRAIN}_metric"].metric._state.num_predictions == 0
    assert model.metrics[f"{ct.EVAL}_metric"].metric._state.num_predictions == batch_size


def test_attach_without_metric() -> None:
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
    )
    engine = Mock()
    model.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_attach_with_metric() -> None:
    engine = create_dummy_engine()
    model = VanillaModel(
        network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
        criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
        metrics=nn.ModuleDict(
            {
                f"{ct.TRAIN}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.TRAIN)),
                f"{ct.EVAL}_metric": VanillaMetric(CategoricalAccuracy(mode=ct.EVAL)),
            }
        ),
    )
    model.attach(engine)
    assert engine.has_event_handler(
        VanillaEventHandler(model.metrics[f"{ct.TRAIN}_metric"].metric.reset),
        event=EngineEvents.TRAIN_EPOCH_STARTED,
    )
    assert engine.has_event_handler(
        VanillaEventHandler(
            model.metrics[f"{ct.TRAIN}_metric"].metric.value, handler_kwargs={"engine": engine}
        ),
        event=EngineEvents.TRAIN_EPOCH_COMPLETED,
    )
    assert engine.has_event_handler(
        VanillaEventHandler(model.metrics[f"{ct.EVAL}_metric"].metric.reset),
        event=EngineEvents.EVAL_EPOCH_STARTED,
    )
    assert engine.has_event_handler(
        VanillaEventHandler(
            model.metrics[f"{ct.EVAL}_metric"].metric.value, handler_kwargs={"engine": engine}
        ),
        event=EngineEvents.EVAL_EPOCH_COMPLETED,
    )


def test_vanilla_model_no_checkpoint_path() -> None:
    with patch("gravitorch.models.vanilla.load_checkpoint_to_module") as load_mock:
        VanillaModel(
            network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
            criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
            checkpoint_path=None,
        )
        load_mock.assert_not_called()


def test_vanilla_model_checkpoint_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    with patch("gravitorch.models.vanilla.load_checkpoint_to_module") as load_mock:
        model = VanillaModel(
            network=BetaMLP(input_size=16, hidden_sizes=(32, 8)),
            criterion=VanillaLoss(criterion=nn.CrossEntropyLoss()),
            checkpoint_path=checkpoint_path,
        )
        load_mock.assert_called_once_with(checkpoint_path, model)


def test_vanilla_model_parse_net_out_torch_tensor() -> None:
    network = Mock()
    network.get_output_names.return_value = ("name1",)
    model = VanillaModel(network=network, criterion=Mock())
    assert objects_are_equal(model._parse_net_out(torch.ones(2, 3)), {"name1": torch.ones(2, 3)})


def test_vanilla_model_parse_net_out_packed_sequence() -> None:
    network = Mock()
    network.get_output_names.return_value = ("name1",)
    model = VanillaModel(network=network, criterion=Mock())
    out = pack_sequence([torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])])
    assert objects_are_equal(model._parse_net_out(out), {"name1": out})


def test_vanilla_model_parse_net_out_tuple_1() -> None:
    network = Mock()
    network.get_output_names.return_value = ("name1",)
    model = VanillaModel(network=network, criterion=Mock())
    assert objects_are_equal(model._parse_net_out((torch.ones(2, 3),)), {"name1": torch.ones(2, 3)})


def test_vanilla_model_parse_net_out_tuple_2() -> None:
    network = Mock()
    network.get_output_names.return_value = ("name1", "name2")
    model = VanillaModel(network=network, criterion=Mock())
    assert objects_are_equal(
        model._parse_net_out(
            (
                torch.ones(2, 3),
                torch.zeros(2, 3, 4),
            )
        ),
        {"name1": torch.ones(2, 3), "name2": torch.zeros(2, 3, 4)},
    )
