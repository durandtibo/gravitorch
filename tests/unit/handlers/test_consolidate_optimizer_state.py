import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark
from torch import nn
from torch.optim import SGD

from gravitorch.engines import EngineEvents
from gravitorch.handlers import ConsolidateOptimizerStateHandler
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


######################################################
#     Tests for ConsolidateOptimizerStateHandler     #
######################################################


def test_consolidate_optimizer_state_handler_str():
    assert str(ConsolidateOptimizerStateHandler()).startswith("ConsolidateOptimizerStateHandler(")


@mark.parametrize("event", EVENTS)
def test_consolidate_optimizer_state_handler_event(event: str):
    assert ConsolidateOptimizerStateHandler(event)._event == event


def test_consolidate_optimizer_state_handler_event_default():
    assert ConsolidateOptimizerStateHandler()._event == EngineEvents.TRAIN_EPOCH_COMPLETED


@mark.parametrize("recipient_rank", (-1, 0, 1))
def test_consolidate_optimizer_state_dict_recipient_rank(recipient_rank: int):
    assert ConsolidateOptimizerStateHandler()._recipient_rank == 0


def test_consolidate_optimizer_state_dict_recipient_rank_default():
    assert ConsolidateOptimizerStateHandler()._recipient_rank == 0


@mark.parametrize("event", EVENTS)
def test_consolidate_optimizer_state_handler_attach(event: str):
    handler = ConsolidateOptimizerStateHandler(event=event)
    engine = Mock(epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(
            handler.consolidate,
            handler_kwargs={"engine": engine},
        ),
    )


def test_consolidate_optimizer_state_handler_attach_duplicate():
    engine = Mock(epoch=-1, has_event_handler=Mock(return_value=True))
    ConsolidateOptimizerStateHandler().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_consolidate_optimizer_state_handler_consolidate_no_optimizer(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        ConsolidateOptimizerStateHandler().consolidate(engine=Mock(optimizer=None))
        assert len(caplog.messages) == 1


@mark.parametrize("recipient_rank", (0, 1))
def test_consolidate_optimizer_state_handler_consolidate(recipient_rank: int):
    engine = Mock()
    ConsolidateOptimizerStateHandler(recipient_rank=recipient_rank).consolidate(engine)
    engine.optimizer.consolidate_state_dict.assert_called_once_with(recipient_rank)


def test_consolidate_optimizer_state_handler_consolidate_no_consolidate_state_dict(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        ConsolidateOptimizerStateHandler().consolidate(
            engine=Mock(optimizer=SGD(nn.Linear(4, 6).parameters(), lr=0.01))
        )
        assert len(caplog.messages) == 1
