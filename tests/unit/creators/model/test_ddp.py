import logging
from typing import Union
from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import LogCaptureFixture, fixture, mark
from torch import nn

from gravitorch import distributed as dist
from gravitorch.creators.model import (
    BaseModelCreator,
    DataDistributedParallelModelCreator,
    ModelCreator,
)
from gravitorch.creators.model.ddp import to_ddp
from gravitorch.engines import BaseEngine

#########################################################
#     Tests for DataDistributedParallelModelCreator     #
#########################################################


@fixture
def model_creator() -> BaseModelCreator:
    return ModelCreator(
        model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
    )


def test_data_distributed_parallel_model_creator_str() -> None:
    assert str(DataDistributedParallelModelCreator(model_creator=Mock())).startswith(
        "DataDistributedParallelModelCreator("
    )


@mark.parametrize(
    "model_creator",
    (
        ModelCreator(
            model_config={OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6}
        ),
        {
            OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
            "model_config": {OBJECT_TARGET: "torch.nn.Linear", "in_features": 4, "out_features": 6},
        },
    ),
)
def test_data_distributed_parallel_model_creator_model_creator(
    model_creator: Union[BaseModelCreator, dict]
) -> None:
    assert isinstance(
        DataDistributedParallelModelCreator(model_creator=model_creator)._model_creator,
        ModelCreator,
    )


def test_data_distributed_parallel_model_creator_ddp_kwargs_default(
    model_creator: BaseModelCreator,
) -> None:
    assert DataDistributedParallelModelCreator(model_creator=model_creator)._ddp_kwargs == {}


def test_data_distributed_parallel_model_creator_ddp_kwargs(
    model_creator: BaseModelCreator,
) -> None:
    assert DataDistributedParallelModelCreator(
        model_creator=model_creator, ddp_kwargs={"find_unused_parameters": True}
    )._ddp_kwargs == {"find_unused_parameters": True}


def test_data_distributed_parallel_model_creator_create() -> None:
    model = nn.Linear(4, 6)
    model_creator = Mock()
    model_creator.create.return_value = model
    creator = DataDistributedParallelModelCreator(model_creator=model_creator)
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.to_ddp", ddp_mock):
        creator.create(engine=Mock(spec=BaseEngine))
        ddp_mock.assert_called_once_with(module=model, ddp_kwargs={})


def test_data_distributed_parallel_model_creator_create_ddp_kwargs() -> None:
    model = nn.Linear(4, 6)
    model_creator = Mock()
    model_creator.create.return_value = model
    creator = DataDistributedParallelModelCreator(
        model_creator=model_creator,
        ddp_kwargs={"find_unused_parameters": True},
    )
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.to_ddp", ddp_mock):
        creator.create(engine=Mock(spec=BaseEngine))
        ddp_mock.assert_called_once_with(module=model, ddp_kwargs={"find_unused_parameters": True})


############################
#     Tests for to_ddp     #
############################


@patch("gravitorch.creators.model.ddp.isinstance", lambda *args, **kwargs: True)
def test_to_ddp_already_ddp(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        module = nn.Linear(4, 5)
        assert to_ddp(module) is module
        assert len(caplog.messages) >= 1


@patch("gravitorch.creators.model.ddp.isinstance", lambda *args, **kwargs: False)
@patch("gravitorch.creators.model.ddp.dist.backend", lambda *args, **kwargs: dist.Backend.GLOO)
def test_to_ddp_gloo() -> None:
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module)
        ddp_mock.assert_called_once_with(module)


@patch("gravitorch.creators.model.ddp.isinstance", lambda *args, **kwargs: False)
@patch("gravitorch.creators.model.ddp.dist.backend", lambda *args, **kwargs: dist.Backend.GLOO)
def test_to_ddp_gloo_ddp_kwargs() -> None:
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module, ddp_kwargs={"find_unused_parameters": True})
        ddp_mock.assert_called_once_with(module, find_unused_parameters=True)


@patch("gravitorch.creators.model.ddp.isinstance", lambda *args, **kwargs: False)
@patch("gravitorch.creators.model.ddp.dist.backend", lambda *args, **kwargs: dist.Backend.NCCL)
@patch("gravitorch.creators.model.ddp.dist.get_local_rank", lambda *args, **kwargs: 1)
def test_to_ddp_nccl() -> None:
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module)
        ddp_mock.assert_called_once_with(module, device_ids=[1])


@patch("gravitorch.creators.model.ddp.isinstance", lambda *args, **kwargs: False)
@patch("gravitorch.creators.model.ddp.dist.backend", lambda *args, **kwargs: dist.Backend.NCCL)
@patch("gravitorch.creators.model.ddp.dist.get_local_rank", lambda *args, **kwargs: 1)
def test_to_ddp_nccl_ddp_kwargs() -> None:
    ddp_mock = Mock()
    with patch("gravitorch.creators.model.ddp.DistributedDataParallel", ddp_mock):
        module = nn.Linear(4, 5)
        to_ddp(module, ddp_kwargs={"find_unused_parameters": True})
        ddp_mock.assert_called_once_with(module, device_ids=[1], find_unused_parameters=True)


@patch("gravitorch.creators.model.ddp.dist.backend", lambda *args, **kwargs: "UNKNOWN_BACKEND")
def test_to_ddp_unknown_backend() -> None:
    assert isinstance(to_ddp(nn.Linear(4, 6)), nn.Linear)
