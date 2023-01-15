from unittest.mock import Mock, patch

from objectory import OBJECT_TARGET
from pytest import mark
from torch import nn

from gravitorch import constants as ct
from gravitorch.creators.optimizer import ZeroRedundancyOptimizerCreator
from gravitorch.engines import BaseEngine

####################################################
#     Tests for ZeroRedundancyOptimizerCreator     #
####################################################


def test_zero_redundancy_optimizer_creator_str():
    assert str(
        ZeroRedundancyOptimizerCreator(
            optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}
        )
    ).startswith("ZeroRedundancyOptimizerCreator(")


@mark.parametrize("lr", (0.01, 0.001))
def test_zero_redundancy_optimizer_creator_create_optimizer_config(lr: float):
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": lr}
    )
    zero_mock = Mock()
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", zero_mock):
        creator.create(engine=Mock(), model=nn.Linear(4, 6))
        assert zero_mock.call_args.kwargs["lr"] == lr


def test_zero_redundancy_optimizer_creator_create_zero_kwargs():
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        zero_kwargs={"overlap_with_ddp": False},
    )
    zero_mock = Mock()
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", zero_mock):
        creator.create(engine=Mock(), model=nn.Linear(4, 6))
        assert not zero_mock.call_args.kwargs["overlap_with_ddp"]


def test_zero_redundancy_optimizer_creator_create_add_module_to_engine_true():
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
    )
    engine = Mock()
    zero_mock = Mock()
    zero_mock.return_value = "my_optimizer"
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", zero_mock):
        creator.create(engine=engine, model=nn.Linear(4, 6))
        engine.add_module.assert_called_once_with(ct.OPTIMIZER, "my_optimizer")


def test_zero_redundancy_optimizer_creator_create_add_module_to_engine_false():
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        add_module_to_engine=False,
    )
    engine = Mock()
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", Mock()):
        creator.create(engine=engine, model=nn.Linear(4, 6))
        engine.add_module.assert_not_called()


def test_zero_redundancy_optimizer_creator_create_attach_handler_true():
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
    )
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", Mock()):
        creator.create(engine=engine, model=nn.Linear(4, 6))
        engine.add_event_handler.assert_called_once()


def test_zero_redundancy_optimizer_creator_create_attach_handler_false():
    creator = ZeroRedundancyOptimizerCreator(
        optimizer_config={OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01},
        attach_handler=False,
    )
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    with patch("gravitorch.creators.optimizer.zero.ZeroRedundancyOptimizer", Mock()):
        creator.create(engine=engine, model=nn.Linear(4, 6))
        engine.add_event_handler.assert_not_called()
