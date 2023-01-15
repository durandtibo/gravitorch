from pytest import fixture

from gravitorch.utils.history import (
    BaseHistory,
    GenericHistory,
    MaxScalarHistory,
    MinScalarHistory,
    get_best_values,
    get_last_values,
)


@fixture
def histories() -> dict[str, BaseHistory]:
    history1 = MinScalarHistory("loss")
    history1.add_value(1.9)
    history1.add_value(1.2)
    history2 = MaxScalarHistory("accuracy")
    history2.add_value(42)
    history2.add_value(35)
    history3 = GenericHistory("epoch")
    history3.add_value(0)
    history3.add_value(1)
    history4 = MaxScalarHistory("f1")
    return {
        history1.name: history1,
        history2.name: history2,
        history3.name: history3,
        history4.name: history4,
    }


#####################################
#     Tests for get_best_values     #
#####################################


def test_get_best_values(histories: dict[str, BaseHistory]):
    assert get_best_values(histories) == {"loss": 1.2, "accuracy": 42}


def test_get_best_values_prefix(histories: dict[str, BaseHistory]):
    assert get_best_values(histories, prefix="best/") == {"best/loss": 1.2, "best/accuracy": 42}


def test_get_best_values_suffix(histories: dict[str, BaseHistory]):
    assert get_best_values(histories, suffix="/best") == {"loss/best": 1.2, "accuracy/best": 42}


def test_get_best_values_empty():
    assert get_best_values({}) == {}


#####################################
#     Tests for get_last_values     #
#####################################


def test_get_last_values(histories: dict[str, BaseHistory]):
    assert get_last_values(histories) == {"loss": 1.2, "accuracy": 35, "epoch": 1}


def test_get_last_values_prefix(histories: dict[str, BaseHistory]):
    assert get_last_values(histories, prefix="last/") == {
        "last/loss": 1.2,
        "last/accuracy": 35,
        "last/epoch": 1,
    }


def test_get_last_values_suffix(histories: dict[str, BaseHistory]):
    assert get_last_values(histories, suffix="/last") == {
        "loss/last": 1.2,
        "accuracy/last": 35,
        "epoch/last": 1,
    }


def test_get_last_values_empty():
    assert get_last_values({}) == {}
