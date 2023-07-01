from unittest.mock import Mock

import torch
from coola import objects_are_allclose, objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch.nn import Linear
from torch.optim import SGD, Adam
from torch.utils.data import Dataset

from gravitorch import constants as ct
from gravitorch.datasources import BaseDataSource
from gravitorch.testing import (
    DummyClassificationModel,
    DummyDataset,
    DummyDataSource,
    DummyIterableDataset,
    create_dummy_engine,
)
from gravitorch.utils import get_available_devices

##################################
#     Tests for DummyDataset     #
##################################


def test_dummy_dataset_str() -> None:
    assert str(DummyDataset()).startswith("DummyDataset(")


def test_dummy_dataset_getitem() -> None:
    dataset = DummyDataset()
    assert objects_are_equal(dataset[0], {ct.INPUT: torch.ones(4), ct.TARGET: 1})
    assert objects_are_equal(dataset[1], {ct.INPUT: torch.full((4,), 2.0), ct.TARGET: 1})
    assert objects_are_equal(dataset[2], {ct.INPUT: torch.full((4,), 3.0), ct.TARGET: 1})


@mark.parametrize("num_examples", (1, 2, 3))
def test_dummy_dataset_len(num_examples: int) -> None:
    assert len(DummyDataset(num_examples=num_examples)) == num_examples


##########################################
#     Tests for DummyIterableDataset     #
##########################################


def test_dummy_iterable_dataset_str() -> None:
    assert str(DummyIterableDataset()).startswith("DummyIterableDataset(")


@mark.parametrize("num_examples", (1, 2, 3))
def test_dummy_iterable_dataset_iter(num_examples: int) -> None:
    assert len(list(DummyIterableDataset(num_examples=num_examples)))


def test_dummy_iterable_dataset_getitem() -> None:
    dataset = iter(DummyIterableDataset())
    assert objects_are_equal(next(dataset), {ct.INPUT: torch.full((4,), 2.0), ct.TARGET: 1})
    assert objects_are_equal(next(dataset), {ct.INPUT: torch.full((4,), 3.0), ct.TARGET: 1})
    assert objects_are_equal(next(dataset), {ct.INPUT: torch.full((4,), 4.0), ct.TARGET: 1})


def test_dummy_iterable_dataset_len_has_length_false() -> None:
    with raises(TypeError, match="DummyIterableDataset instance doesn't have valid length"):
        len(DummyIterableDataset())


@mark.parametrize("num_examples", (1, 2, 3))
def test_dummy_iterable_dataset_len_has_length_true(num_examples: int) -> None:
    assert len(DummyIterableDataset(num_examples=num_examples, has_length=True)) == num_examples


#####################################
#     Tests for DummyDataSource     #
#####################################


def test_dummy_datasource_str() -> None:
    assert str(DummyDataSource()).startswith("DummyDataSource(")


def test_dummy_datasource_default_datasets() -> None:
    datasource = DummyDataSource()
    assert len(datasource._datasets) == 2
    assert isinstance(datasource._datasets[ct.TRAIN], DummyDataset)
    assert isinstance(datasource._datasets[ct.EVAL], DummyDataset)


def test_dummy_datasource_datasets() -> None:
    train_dataset = Mock(spec=Dataset)
    eval_dataset = Mock(spec=Dataset)
    datasource = DummyDataSource(train_dataset=train_dataset, eval_dataset=eval_dataset)
    assert len(datasource._datasets) == 2
    assert datasource._datasets[ct.TRAIN] is train_dataset
    assert datasource._datasets[ct.EVAL] is eval_dataset


##############################################
#     Tests for DummyClassificationModel     #
##############################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", (1, 2))
@mark.parametrize("feature_size", (1, 2))
def test_dummy_classification_model_forward(
    device: str, batch_size: int, feature_size: int
) -> None:
    device = torch.device(device)
    model = DummyClassificationModel(feature_size=feature_size).to(device=device)
    out = model(
        {
            ct.INPUT: torch.rand(batch_size, feature_size, device=device),
            ct.TARGET: torch.zeros(batch_size, dtype=torch.long, device=device),
        }
    )
    assert len(out) == 1
    assert torch.is_tensor(out[ct.LOSS])
    assert out[ct.LOSS].shape == ()
    assert out[ct.LOSS].dtype == torch.float
    assert out[ct.LOSS].device == device


def test_dummy_classification_model_nan() -> None:
    model = DummyClassificationModel(loss_nan=True)
    assert objects_are_allclose(
        model({ct.INPUT: torch.rand(2, 4), ct.TARGET: torch.zeros(2, dtype=torch.long)}),
        {ct.LOSS: torch.tensor(float("nan"))},
        equal_nan=True,
    )


#########################################
#     Tests for create_dummy_engine     #
#########################################


def test_create_dummy_engine_default() -> None:
    engine = create_dummy_engine()
    assert isinstance(engine.datasource, DummyDataSource)
    assert isinstance(engine.model, DummyClassificationModel)
    assert isinstance(engine.optimizer, SGD)


def test_create_dummy_engine() -> None:
    datasource = Mock(spec=BaseDataSource)
    engine = create_dummy_engine(
        datasource=datasource,
        model=Linear(4, 6),
        optimizer={OBJECT_TARGET: "torch.optim.Adam", "lr": 0.0003},
    )
    assert engine.datasource is datasource
    assert isinstance(engine.model, Linear)
    assert isinstance(engine.optimizer, Adam)
