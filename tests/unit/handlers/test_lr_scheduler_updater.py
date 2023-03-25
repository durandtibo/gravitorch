from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.events import VanillaEventHandler
from gravitorch.handlers import (
    EpochLRSchedulerUpdater,
    IterationLRSchedulerUpdater,
    LRSchedulerUpdater,
    MetricEpochLRSchedulerUpdater,
    MetricLRSchedulerUpdater,
)
from gravitorch.utils.history import MinScalarHistory

EVENTS = ("my_event", "my_other_event")
METRICS = ("metric1", "metric2")


########################################
#     Tests for LRSchedulerUpdater     #
########################################


def test_lr_scheduler_updater_str() -> None:
    assert str(LRSchedulerUpdater("my_event")).startswith("LRSchedulerUpdater(")


@mark.parametrize("event", EVENTS)
def test_lr_scheduler_updater_event(event: str) -> None:
    assert LRSchedulerUpdater(event)._event == event


@mark.parametrize("event", EVENTS)
def test_lr_scheduler_updater_attach(event: str) -> None:
    handler = LRSchedulerUpdater(event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event, VanillaEventHandler(engine.lr_scheduler.step)
    )


def test_lr_scheduler_updater_attach_duplicate() -> None:
    handler = LRSchedulerUpdater("my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_lr_scheduler_updater_attach_lr_scheduler_none() -> None:
    handler = LRSchedulerUpdater("my_event")
    engine = Mock(spec=BaseEngine, lr_scheduler=None)
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_epoch_lr_scheduler_updater_event() -> None:
    assert EpochLRSchedulerUpdater()._event == EngineEvents.TRAIN_EPOCH_COMPLETED


def test_iteration_lr_scheduler_updater_event() -> None:
    assert IterationLRSchedulerUpdater()._event == EngineEvents.TRAIN_ITERATION_COMPLETED


##############################################
#     Tests for MetricLRSchedulerUpdater     #
##############################################


def test_metric_lr_scheduler_updater_str() -> None:
    assert str(MetricLRSchedulerUpdater("my_event")).startswith("MetricLRSchedulerUpdater(")


@mark.parametrize("event", EVENTS)
def test_metric_lr_scheduler_updater_event(event: str) -> None:
    assert MetricLRSchedulerUpdater(event)._event == event


@mark.parametrize("metric_name", METRICS)
def test_metric_lr_scheduler_updater_metric_name(metric_name: str) -> None:
    assert MetricLRSchedulerUpdater("my_event", metric_name)._metric_name == metric_name


@mark.parametrize("event", EVENTS)
def test_metric_lr_scheduler_updater_attach(event: str) -> None:
    handler = MetricLRSchedulerUpdater(event=event)
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=False))
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event, VanillaEventHandler(handler.step, handler_kwargs={"engine": engine})
    )


def test_metric_lr_scheduler_updater_attach_duplicate() -> None:
    handler = MetricLRSchedulerUpdater("my_event")
    engine = Mock(spec=BaseEngine, has_event_handler=Mock(return_value=True))
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


def test_metric_lr_scheduler_updater_attach_lr_scheduler_none() -> None:
    handler = MetricLRSchedulerUpdater("my_event")
    engine = Mock(spec=BaseEngine, lr_scheduler=None)
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


@mark.parametrize("metric_name", METRICS)
def test_metric_lr_scheduler_updater_step(metric_name: str) -> None:
    handler = MetricLRSchedulerUpdater("my_event", metric_name)
    engine = Mock(
        spec=BaseEngine,
        get_history=Mock(return_value=MinScalarHistory(metric_name, elements=((None, 1.2),))),
    )
    handler.step(engine)
    engine.lr_scheduler.step.assert_called_once_with(1.2)


###################################################
#     Tests for MetricEpochLRSchedulerUpdater     #
###################################################


def test_epoch_lr_scheduler_with_metric_updater_event() -> None:
    assert MetricEpochLRSchedulerUpdater()._event == EngineEvents.EPOCH_COMPLETED
