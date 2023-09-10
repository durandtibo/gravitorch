from unittest.mock import Mock

import torch
from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.model import ModelCreator
from gravitorch.engines import BaseEngine
from gravitorch.nn import get_module_device
from gravitorch.testing import cuda_available
from gravitorch.utils.device_placement import (
    AutoDevicePlacement,
    CpuDevicePlacement,
    CudaDevicePlacement,
)

##################################
#     Tests for ModelCreator     #
##################################


def test_model_creator_str() -> None:
    assert str(ModelCreator(model_config={})).startswith("ModelCreator(")


@mark.parametrize("attach_model_to_engine", (True, False))
def test_model_creator_attach_model_to_engine(attach_model_to_engine: bool) -> None:
    assert (
        ModelCreator(
            model_config={}, attach_model_to_engine=attach_model_to_engine
        )._attach_model_to_engine
        == attach_model_to_engine
    )


@mark.parametrize("add_module_to_engine", (True, False))
def test_model_creator_add_module_to_engine(add_module_to_engine: bool) -> None:
    assert (
        ModelCreator(
            model_config={}, add_module_to_engine=add_module_to_engine
        )._add_module_to_engine
        == add_module_to_engine
    )


def test_model_creator_device_placement_default() -> None:
    assert isinstance(ModelCreator(model_config={})._device_placement, AutoDevicePlacement)


def test_model_creator_device_placement_cpu() -> None:
    assert isinstance(
        ModelCreator(model_config={}, device_placement=CpuDevicePlacement())._device_placement,
        CpuDevicePlacement,
    )


def test_model_creator_create_attach_model_to_engine_true() -> None:
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2}
    )
    model = creator.create(engine=Mock(spec=BaseEngine))
    assert isinstance(model, nn.Linear)


def test_model_creator_create_attach_model_to_engine_false() -> None:
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
        attach_model_to_engine=False,
    )
    model = creator.create(engine=Mock(spec=BaseEngine))
    assert isinstance(model, nn.Linear)


def test_model_creator_create_add_module_to_engine_true() -> None:
    engine = Mock(spec=BaseEngine)
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2}
    )
    model = creator.create(engine=engine)
    assert isinstance(model, nn.Linear)
    engine.add_module.assert_called_once_with(ct.MODEL, model)


def test_model_creator_create_add_module_to_engine_false() -> None:
    engine = Mock(spec=BaseEngine)
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
        add_module_to_engine=False,
    )
    model = creator.create(engine=engine)
    assert isinstance(model, nn.Linear)
    engine.add_module.assert_not_called()


def test_model_creator_create_device_placement_cpu() -> None:
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
        device_placement=CpuDevicePlacement(),
    )
    model = creator.create(engine=Mock(spec=BaseEngine))
    assert isinstance(model, nn.Linear)
    assert get_module_device(model) == torch.device("cpu")


@cuda_available
def test_model_creator_create_device_placement_cuda() -> None:
    creator = ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 8, "out_features": 2},
        device_placement=CudaDevicePlacement(),
    )
    model = creator.create(engine=Mock(spec=BaseEngine))
    assert isinstance(model, nn.Linear)
    assert get_module_device(model) == torch.device("cuda:0")
