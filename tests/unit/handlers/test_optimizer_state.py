import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark
from torch import nn
from torch.optim import SGD

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.handlers import ConsolidateOptimizerState
from gravitorch.utils.events import VanillaEventHandler

EVENTS = ("my_event", "my_other_event")


###############################################
#     Tests for ConsolidateOptimizerState     #
###############################################


def test_consolidate_optimizer_state_str() -> None:
    assert str(ConsolidateOptimizerState()).startswith("ConsolidateOptimizerState(")


@mark.parametrize("event", EVENTS)
def test_consolidate_optimizer_state_event(event: str) -> None:
    assert ConsolidateOptimizerState(event)._event == event


def test_consolidate_optimizer_state_event_default() -> None:
    assert ConsolidateOptimizerState()._event == EngineEvents.TRAIN_EPOCH_COMPLETED


@mark.parametrize("recipient_rank", (-1, 0, 1))
def test_consolidate_optimizer_state_dict_recipient_rank(recipient_rank: int) -> None:
    assert ConsolidateOptimizerState()._recipient_rank == 0


def test_consolidate_optimizer_state_dict_recipient_rank_default() -> None:
    assert ConsolidateOptimizerState()._recipient_rank == 0


@mark.parametrize("event", EVENTS)
def test_consolidate_optimizer_state_attach(event: str) -> None:
    handler = ConsolidateOptimizerState(event=event)
    engine = Mock(epoch=-1, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        VanillaEventHandler(
            handler.consolidate,
            handler_kwargs={"engine": engine},
        ),
    )


def test_consolidate_optimizer_state_attach_duplicate() -> None:
    engine = Mock(epoch=-1, has_event_handler=Mock(return_value=True))
    ConsolidateOptimizerState().attach(engine)
    engine.add_event_handler.assert_not_called()


def test_consolidate_optimizer_state_consolidate_no_optimizer(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        ConsolidateOptimizerState().consolidate(engine=Mock(optimizer=None))
        assert len(caplog.messages) == 1


@mark.parametrize("recipient_rank", (0, 1))
def test_consolidate_optimizer_state_consolidate(recipient_rank: int) -> None:
    engine = Mock(spec=BaseEngine)
    ConsolidateOptimizerState(recipient_rank=recipient_rank).consolidate(engine)
    engine.optimizer.consolidate_state_dict.assert_called_once_with(recipient_rank)


def test_consolidate_optimizer_state_consolidate_no_consolidate_state_dict(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        ConsolidateOptimizerState().consolidate(
            engine=Mock(spec=BaseEngine, optimizer=SGD(nn.Linear(4, 6).parameters(), lr=0.01))
        )
        assert len(caplog.messages) == 1
