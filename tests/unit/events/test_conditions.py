from unittest.mock import Mock

from pytest import mark

from gravitorch.engines import BaseEngine
from gravitorch.events import EpochPeriodicCondition, IterationPeriodicCondition

############################################
#     Tests for EpochPeriodicCondition     #
############################################


def test_epoch_periodic_condition_str() -> None:
    engine = Mock(spec=BaseEngine)
    engine.epoch = -1
    assert str(EpochPeriodicCondition(engine, freq=2)).startswith("EpochPeriodicCondition(freq=2,")


@mark.parametrize("freq", (1, 2, 3))
def test_epoch_periodic_condition_freq(freq: int) -> None:
    assert EpochPeriodicCondition(engine=Mock(spec=BaseEngine), freq=freq).freq == freq


def test_epoch_periodic_condition_eq_true() -> None:
    assert EpochPeriodicCondition(engine=Mock(spec=BaseEngine), freq=3) == EpochPeriodicCondition(
        engine=Mock(spec=BaseEngine), freq=3
    )


def test_epoch_periodic_condition_eq_false_different_freq() -> None:
    assert EpochPeriodicCondition(engine=Mock(spec=BaseEngine), freq=3) != EpochPeriodicCondition(
        engine=Mock(spec=BaseEngine), freq=2
    )


def test_epoch_periodic_condition_eq_false_different_classes() -> None:
    assert EpochPeriodicCondition(engine=Mock(spec=BaseEngine), freq=3) != "meow"


def test_epoch_periodic_condition_true() -> None:
    engine = Mock(spec=BaseEngine)
    condition = EpochPeriodicCondition(engine, 2)
    engine.epoch = 0
    assert condition()
    engine.epoch = 2
    assert condition()


def test_epoch_periodic_condition_false() -> None:
    engine = Mock(spec=BaseEngine)
    condition = EpochPeriodicCondition(engine, 2)
    engine.epoch = -1
    assert not condition()
    engine.epoch = 1
    assert not condition()


################################################
#     Tests for IterationPeriodicCondition     #
################################################


def test_iteration_periodic_condition_str() -> None:
    engine = Mock(spec=BaseEngine)
    engine.iteration = -1
    assert str(IterationPeriodicCondition(engine, freq=2)).startswith(
        "IterationPeriodicCondition(freq=2,"
    )


@mark.parametrize("freq", (1, 2, 3))
def test_iteration_periodic_condition_freq(freq: int) -> None:
    assert IterationPeriodicCondition(engine=Mock(spec=BaseEngine), freq=freq).freq == freq


def test_iteration_periodic_condition_eq_true() -> None:
    assert IterationPeriodicCondition(
        engine=Mock(spec=BaseEngine), freq=3
    ) == IterationPeriodicCondition(engine=Mock(spec=BaseEngine), freq=3)


def test_iteration_periodic_condition_eq_false_different_freq() -> None:
    assert IterationPeriodicCondition(
        engine=Mock(spec=BaseEngine), freq=3
    ) != IterationPeriodicCondition(engine=Mock(spec=BaseEngine), freq=2)


def test_iteration_periodic_condition_eq_false_different_classes() -> None:
    assert IterationPeriodicCondition(engine=Mock(spec=BaseEngine), freq=3) != "meow"


def test_iteration_periodic_condition_true() -> None:
    engine = Mock(spec=BaseEngine)
    condition = IterationPeriodicCondition(engine, 2)
    engine.iteration = 0
    assert condition()
    engine.iteration = 2
    assert condition()


def test_iteration_periodic_condition_false() -> None:
    engine = Mock(spec=BaseEngine)
    condition = IterationPeriodicCondition(engine, 2)
    engine.iteration = -1
    assert not condition()
    engine.iteration = 1
    assert not condition()
