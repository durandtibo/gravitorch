from unittest.mock import Mock, patch

from pytest import mark, raises

from gravitorch.engines import EngineEvents
from gravitorch.handlers import EpochOptimizerMonitor, IterationOptimizerMonitor
from gravitorch.utils.events import (
    ConditionalEventHandler,
    EpochPeriodicCondition,
    IterationPeriodicCondition,
)
from gravitorch.utils.exp_trackers import EpochStep, IterationStep

EVENTS = ("my_event", "my_other_event")


###########################################
#     Tests for EpochOptimizerMonitor     #
###########################################


def test_epoch_optimizer_monitor_str():
    assert str(EpochOptimizerMonitor()).startswith("EpochOptimizerMonitor(")


@mark.parametrize("event", EVENTS)
def test_epoch_optimizer_monitor_event(event: str):
    assert EpochOptimizerMonitor(event)._event == event


def test_epoch_optimizer_monitor_event_default():
    assert EpochOptimizerMonitor()._event == EngineEvents.TRAIN_EPOCH_STARTED


@mark.parametrize("freq", (1, 2))
def test_epoch_optimizer_monitor_freq(freq: int):
    assert EpochOptimizerMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_epoch_optimizer_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        EpochOptimizerMonitor(freq=freq)


def test_epoch_optimizer_monitor_freq_default():
    assert EpochOptimizerMonitor()._freq == 1


@mark.parametrize("tablefmt", ("fancy_grid", "github"))
def test_epoch_optimizer_monitor_tablefmt(tablefmt: str):
    assert EpochOptimizerMonitor(tablefmt=tablefmt)._tablefmt == tablefmt


def test_epoch_optimizer_monitor_tablefmt_default():
    assert EpochOptimizerMonitor()._tablefmt == "fancy_grid"


@mark.parametrize("prefix", ("train/", "eval/"))
def test_epoch_optimizer_monitor_prefix(prefix: str):
    assert EpochOptimizerMonitor(prefix=prefix)._prefix == prefix


def test_epoch_optimizer_monitor_prefix_default():
    assert EpochOptimizerMonitor()._prefix == "train/"


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_epoch_optimizer_monitor_attach(event: str, freq: int):
    handler = EpochOptimizerMonitor(event=event, freq=freq)
    engine = Mock()
    engine.epoch = -1
    engine.has_event_handler.return_value = False
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=EpochPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_epoch_optimizer_monitor_attach_duplicate():
    handler = EpochOptimizerMonitor()
    engine = Mock()
    engine.epoch = -1
    engine.has_event_handler.return_value = True
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


@mark.parametrize("tablefmt", ("fancy_grid", "github"))
@mark.parametrize("prefix", ("train/", "eval/"))
def test_epoch_optimizer_monitor_monitor(tablefmt: str, prefix: str):
    optimizer_monitor = EpochOptimizerMonitor(tablefmt=tablefmt, prefix=prefix)
    engine = Mock()
    engine.epoch = 1
    with patch(
        "gravitorch.handlers.optimizer_monitor.show_optimizer_parameters_per_group"
    ) as show_mock:
        with patch(
            "gravitorch.handlers.optimizer_monitor.log_optimizer_parameters_per_group"
        ) as log_mock:
            optimizer_monitor.monitor(engine)
            show_mock.assert_called_once_with(optimizer=engine.optimizer, tablefmt=tablefmt)
            log_mock.assert_called_once_with(
                optimizer=engine.optimizer,
                engine=engine,
                step=EpochStep(1),
                prefix=prefix,
            )


def test_epoch_optimizer_monitor_monitor_no_optimizer():
    optimizer_monitor = EpochOptimizerMonitor()
    engine = Mock()
    engine.optimizer = None
    with patch(
        "gravitorch.handlers.optimizer_monitor.show_optimizer_parameters_per_group"
    ) as show_mock:
        with patch(
            "gravitorch.handlers.optimizer_monitor.log_optimizer_parameters_per_group"
        ) as log_mock:
            optimizer_monitor.monitor(engine)
            show_mock.assert_not_called()
            log_mock.assert_not_called()


###############################################
#     Tests for IterationOptimizerMonitor     #
###############################################


def test_iteration_optimizer_monitor_str():
    assert str(IterationOptimizerMonitor()).startswith("IterationOptimizerMonitor(")


@mark.parametrize("event", EVENTS)
def test_iteration_optimizer_monitor_event(event: str):
    assert IterationOptimizerMonitor(event)._event == event


def test_iteration_optimizer_monitor_event_default():
    assert IterationOptimizerMonitor()._event == EngineEvents.TRAIN_ITERATION_STARTED


@mark.parametrize("freq", (1, 2))
def test_iteration_optimizer_monitor_freq(freq: int):
    assert IterationOptimizerMonitor(freq=freq)._freq == freq


@mark.parametrize("freq", (0, -1))
def test_iteration_optimizer_monitor_incorrect_freq(freq: int):
    with raises(ValueError):
        IterationOptimizerMonitor(freq=freq)


def test_iteration_optimizer_monitor_freq_default():
    assert IterationOptimizerMonitor()._freq == 10


@mark.parametrize("tablefmt", ("fancy_grid", "github"))
def test_iteration_optimizer_monitor_tablefmt(tablefmt: str):
    assert IterationOptimizerMonitor(tablefmt=tablefmt)._tablefmt == tablefmt


def test_iteration_optimizer_monitor_tablefmt_default():
    assert IterationOptimizerMonitor()._tablefmt == "fancy_grid"


@mark.parametrize("prefix", ("train/", "eval/"))
def test_iteration_optimizer_monitor_prefix(prefix: str):
    assert IterationOptimizerMonitor(prefix=prefix)._prefix == prefix


def test_iteration_optimizer_monitor_prefix_default():
    assert IterationOptimizerMonitor()._prefix == "train/"


@mark.parametrize("event", EVENTS)
@mark.parametrize("freq", (1, 2))
def test_iteration_optimizer_monitor_attach(event: str, freq: int):
    handler = IterationOptimizerMonitor(event=event, freq=freq)
    engine = Mock()
    engine.iteration = -1
    engine.has_event_handler.return_value = False
    handler.attach(engine)
    engine.add_event_handler.assert_called_once_with(
        event,
        ConditionalEventHandler(
            handler.monitor,
            condition=IterationPeriodicCondition(engine=engine, freq=freq),
            handler_kwargs={"engine": engine},
        ),
    )


def test_iteration_optimizer_monitor_attach_duplicate():
    handler = IterationOptimizerMonitor()
    engine = Mock()
    engine.iteration = -1
    engine.has_event_handler.return_value = True
    handler.attach(engine)
    engine.add_event_handler.assert_not_called()


@mark.parametrize("tablefmt", ("fancy_grid", "github"))
@mark.parametrize("prefix", ("train/", "eval/"))
def test_iteration_optimizer_monitor_monitor(tablefmt: str, prefix: str):
    optimizer_monitor = IterationOptimizerMonitor(tablefmt=tablefmt, prefix=prefix)
    engine = Mock()
    engine.iteration = 10
    with patch(
        "gravitorch.handlers.optimizer_monitor.show_optimizer_parameters_per_group"
    ) as show_mock:
        with patch(
            "gravitorch.handlers.optimizer_monitor.log_optimizer_parameters_per_group"
        ) as log_mock:
            optimizer_monitor.monitor(engine)
            show_mock.assert_called_once_with(optimizer=engine.optimizer, tablefmt=tablefmt)
            log_mock.assert_called_once_with(
                optimizer=engine.optimizer,
                engine=engine,
                step=IterationStep(10),
                prefix=prefix,
            )


def test_iteration_optimizer_monitor_monitor_no_optimizer():
    optimizer_monitor = IterationOptimizerMonitor()
    engine = Mock()
    engine.optimizer = None
    with patch(
        "gravitorch.handlers.optimizer_monitor.show_optimizer_parameters_per_group"
    ) as show_mock:
        with patch(
            "gravitorch.handlers.optimizer_monitor.log_optimizer_parameters_per_group"
        ) as log_mock:
            optimizer_monitor.monitor(engine)
            show_mock.assert_not_called()
            log_mock.assert_not_called()
