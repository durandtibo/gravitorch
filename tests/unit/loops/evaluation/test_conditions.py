from unittest.mock import Mock

from gravitorch.loops.evaluation.conditions import (
    EveryEpochEvalCondition,
    LastEpochEvalCondition,
)

#############################################
#     Tests for EveryEpochEvalCondition     #
#############################################


def test_every_epoch_2_str() -> None:
    assert str(EveryEpochEvalCondition(every=2)) == "EveryEpochEvalCondition(every=2)"


def test_every_epoch_true_every_2_epoch_0() -> None:
    engine = Mock()
    engine.epoch = 0
    condition = EveryEpochEvalCondition(every=2)
    assert condition(engine)


def test_every_epoch_true_every_2_epoch_2() -> None:
    engine = Mock()
    engine.epoch = 2
    condition = EveryEpochEvalCondition(every=2)
    assert condition(engine)


def test_every_epoch_false_every_2_epoch_1() -> None:
    engine = Mock()
    engine.epoch = 1
    condition = EveryEpochEvalCondition(every=2)
    assert not condition(engine)


############################################
#     Tests for LastEpochEvalCondition     #
############################################


def test_last_epoch_2_str() -> None:
    assert str(LastEpochEvalCondition()) == "LastEpochEvalCondition()"


def test_last_epoch_true_max_epochs_1() -> None:
    engine = Mock()
    engine.epoch = 0
    engine.max_epochs = 1
    condition = LastEpochEvalCondition()
    assert condition(engine)


def test_last_epoch_true_max_epochs_10() -> None:
    engine = Mock()
    engine.epoch = 9
    engine.max_epochs = 10
    condition = LastEpochEvalCondition()
    assert condition(engine)


def test_last_epoch_false() -> None:
    engine = Mock()
    engine.epoch = -1
    engine.max_epochs = 1
    condition = LastEpochEvalCondition()
    assert not condition(engine)
