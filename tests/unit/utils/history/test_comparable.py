from numbers import Number

from coola import objects_are_equal
from objectory import OBJECT_TARGET
from pytest import mark, raises

from gravitorch.utils.history.base import BaseHistory, EmptyHistoryError
from gravitorch.utils.history.comparable import (
    ComparableHistory,
    MaxScalarHistory,
    MinScalarHistory,
)
from gravitorch.utils.history.comparator import MaxScalarComparator

#######################################
#     Tests for ComparableHistory     #
#######################################


def test_comparable_history_str_name() -> None:
    assert (
        str(ComparableHistory[Number]("accuracy", comparator=MaxScalarComparator()))
        == "ComparableHistory(name=accuracy, max_size=10, history=())"
    )


@mark.parametrize("name", ("name", "accuracy", ""))
def test_comparable_history_name(name: str) -> None:
    assert ComparableHistory[Number](name, comparator=MaxScalarComparator()).name == name


@mark.parametrize("max_size", (1, 5))
def test_comparable_history_max_size(max_size: int) -> None:
    assert (
        ComparableHistory[Number](
            "accuracy", comparator=MaxScalarComparator(), max_size=max_size
        ).max_size
        == max_size
    )


def test_comparable_history_max_size_incorrect() -> None:
    with raises(ValueError):
        ComparableHistory[Number]("accuracy", MaxScalarComparator(), max_size=0)


def test_comparable_history_add_value() -> None:
    history = ComparableHistory[Number]("accuracy", MaxScalarComparator())
    history.add_value(2)
    history.add_value(4, step=1)
    assert history.equal(
        ComparableHistory[Number](
            "accuracy",
            MaxScalarComparator(),
            elements=((None, 2), (1, 4)),
            best_value=4,
            improved=True,
        ),
    )


def test_comparable_history_get_best_value() -> None:
    assert (
        ComparableHistory[Number](
            "accuracy", comparator=MaxScalarComparator(), elements=[(0, 1)], best_value=4
        ).get_best_value()
        == 4
    )


def test_comparable_history_get_best_value_last_is_best() -> None:
    history = ComparableHistory[Number]("accuracy", comparator=MaxScalarComparator())
    history.add_value(2, step=0)
    history.add_value(4, step=1)
    assert history.get_best_value() == 4


def test_comparable_history_get_best_value_last_is_not_best() -> None:
    history = ComparableHistory[Number]("accuracy", comparator=MaxScalarComparator())
    history.add_value(2, step=0)
    history.add_value(1, step=1)
    assert history.get_best_value() == 2


def test_comparable_history_get_best_value_empty() -> None:
    history = ComparableHistory[Number]("accuracy", MaxScalarComparator())
    with raises(EmptyHistoryError):
        history.get_best_value()


def test_comparable_history_get_recent_history() -> None:
    assert ComparableHistory[Number](
        "accuracy", MaxScalarComparator(), elements=[(1, 123)]
    ).get_recent_history() == ((1, 123),)


def test_comparable_history_get_recent_history_empty() -> None:
    assert ComparableHistory[Number]("accuracy", MaxScalarComparator()).get_recent_history() == ()


def test_comparable_history_get_recent_history_max_size_3() -> None:
    history = ComparableHistory("accuracy", comparator=MaxScalarComparator(), max_size=3)
    for i in range(10):
        history.add_value(100 - i)
    assert history.equal(
        ComparableHistory(
            "accuracy",
            comparator=MaxScalarComparator(),
            max_size=3,
            elements=((None, 93), (None, 92), (None, 91)),
            best_value=100,
            improved=False,
        )
    )


def test_comparable_history_get_last_value() -> None:
    assert (
        ComparableHistory(
            "accuracy",
            comparator=MaxScalarComparator(),
            max_size=3,
            elements=((None, 1), (None, 3)),
        ).get_last_value()
        == 3
    )


def test_comparable_history_get_last_value_empty() -> None:
    history = ComparableHistory("accuracy", comparator=MaxScalarComparator())
    with raises(EmptyHistoryError):
        history.get_last_value()


def test_comparable_history_has_improved_true() -> None:
    history = ComparableHistory[Number]("accuracy", comparator=MaxScalarComparator())
    history.add_value(2, step=0)
    history.add_value(4, step=1)
    assert history.has_improved()


def test_comparable_history_has_improved_false() -> None:
    history = ComparableHistory[Number]("accuracy", comparator=MaxScalarComparator())
    history.add_value(2, step=0)
    history.add_value(1, step=1)
    assert not history.has_improved()


def test_comparable_history_has_improved_empty() -> None:
    history = ComparableHistory("accuracy", comparator=MaxScalarComparator())
    with raises(EmptyHistoryError):
        history.has_improved()


def test_comparable_history_is_better_true() -> None:
    assert ComparableHistory("accuracy", comparator=MaxScalarComparator()).is_better(
        new_value=0.2, old_value=0.1
    )


def test_comparable_history_is_better_false() -> None:
    assert not ComparableHistory("accuracy", comparator=MaxScalarComparator()).is_better(
        new_value=0.1, old_value=0.2
    )


def test_comparable_history_is_comparable() -> None:
    assert ComparableHistory("accuracy", comparator=MaxScalarComparator()).is_comparable()


def test_comparable_history_is_empty_true() -> None:
    assert ComparableHistory("accuracy", comparator=MaxScalarComparator()).is_empty()


def test_comparable_history_is_empty_false() -> None:
    assert not ComparableHistory(
        "accuracy", comparator=MaxScalarComparator(), elements=[(None, 5)]
    ).is_empty()


def test_comparable_history_config_dict() -> None:
    assert objects_are_equal(
        ComparableHistory("accuracy", comparator=MaxScalarComparator()).config_dict(),
        {
            OBJECT_TARGET: "gravitorch.utils.history.comparable.ComparableHistory",
            "name": "accuracy",
            "max_size": 10,
            "comparator": MaxScalarComparator(),
        },
    )


def test_comparable_history_config_dict_create_new_history() -> None:
    assert BaseHistory.factory(
        **ComparableHistory(
            "accuracy", MaxScalarComparator(), max_size=5, elements=[(0, 1)]
        ).config_dict()
    ).equal(ComparableHistory("accuracy", MaxScalarComparator(), max_size=5))


def test_comparable_history_load_state_dict_empty() -> None:
    history = ComparableHistory("accuracy", comparator=MaxScalarComparator())
    history.load_state_dict({"history": (), "improved": False, "best_value": -float("inf")})
    assert history.equal(ComparableHistory("accuracy", comparator=MaxScalarComparator()))


def test_comparable_history_load_state_dict_max_size_2() -> None:
    history = ComparableHistory("accuracy", MaxScalarComparator(), max_size=2)
    history.load_state_dict({"history": ((0, 1), (1, 5)), "improved": True, "best_value": 5})
    history.add_value(7, step=2)
    assert history.equal(
        ComparableHistory(
            "accuracy",
            comparator=MaxScalarComparator(),
            max_size=2,
            elements=((1, 5), (2, 7)),
            best_value=7,
            improved=True,
        )
    )


def test_comparable_history_load_state_dict_reset() -> None:
    history = ComparableHistory("accuracy", MaxScalarComparator(), elements=[(0, 7)])
    history.load_state_dict({"history": ((0, 4), (1, 5)), "improved": True, "best_value": 5})
    assert history.equal(
        ComparableHistory(
            "accuracy",
            comparator=MaxScalarComparator(),
            elements=((0, 4), (1, 5)),
            best_value=5,
            improved=True,
        )
    )


def test_comparable_history_state_dict() -> None:
    assert ComparableHistory(
        "accuracy",
        MaxScalarComparator(),
        elements=((0, 1), (1, 5)),
        best_value=5,
        improved=True,
    ).state_dict() == {
        "history": ((0, 1), (1, 5)),
        "improved": True,
        "best_value": 5,
    }


def test_comparable_history_state_dict_empty() -> None:
    assert ComparableHistory("accuracy", MaxScalarComparator()).state_dict() == {
        "history": (),
        "improved": False,
        "best_value": -float("inf"),
    }


def test_comparable_history_to_dict_from_dict() -> None:
    assert BaseHistory.from_dict(
        ComparableHistory(
            "accuracy",
            MaxScalarComparator(),
            max_size=5,
            elements=[(0, 1)],
            best_value=1,
            improved=True,
        ).to_dict()
    ).equal(
        ComparableHistory(
            "accuracy",
            MaxScalarComparator(),
            max_size=5,
            elements=[(0, 1)],
            best_value=1,
            improved=True,
        )
    )


######################################
#     Tests for MaxScalarHistory     #
######################################


def test_max_scalar_history_equal_true() -> None:
    assert MaxScalarHistory(
        "accuracy", elements=((None, 1.9), (1, 1.2), (2, 0.8)), best_value=0.8, improved=True
    ).equal(
        MaxScalarHistory(
            "accuracy", elements=((None, 1.9), (1, 1.2), (2, 0.8)), best_value=0.8, improved=True
        )
    )


def test_max_scalar_history_equal_true_empty() -> None:
    assert MaxScalarHistory("accuracy").equal(MaxScalarHistory("accuracy"))


def test_max_scalar_history_equal_false_different_values() -> None:
    assert not MaxScalarHistory(
        "accuracy", elements=((None, 1.9), (1, 1.2), (2, 0.8)), best_value=0.8, improved=True
    ).equal(
        MaxScalarHistory(
            "accuracy", elements=((None, 1.9), (1, 1.2), (2, 0.5)), best_value=0.5, improved=True
        )
    )


def test_max_scalar_history_equal_false_different_names() -> None:
    assert not MaxScalarHistory("accuracy").equal(MaxScalarHistory("f1"))


def test_max_scalar_history_equal_false_different_max_sizes() -> None:
    assert not MaxScalarHistory("accuracy").equal(MaxScalarHistory("accuracy", max_size=2))


def test_max_scalar_history_equal_false_different_types() -> None:
    assert not MaxScalarHistory("accuracy").equal(MinScalarHistory("accuracy"))


def test_max_scalar_get_best_value_last_is_best() -> None:
    history = MaxScalarHistory("accuracy")
    history.add_value(2, step=0)
    history.add_value(4, step=1)
    assert history.get_best_value() == 4


def test_max_scalar_get_best_value_last_is_not_best() -> None:
    history = MaxScalarHistory("accuracy")
    history.add_value(2, step=0)
    history.add_value(1, step=1)
    assert history.get_best_value() == 2


def test_max_scalar_to_dict_from_dict() -> None:
    assert BaseHistory.from_dict(
        MaxScalarHistory(
            "accuracy", max_size=5, elements=[(0, 1)], best_value=1, improved=True
        ).to_dict()
    ).equal(
        MaxScalarHistory("accuracy", max_size=5, elements=[(0, 1)], best_value=1, improved=True)
    )


######################################
#     Tests for MinScalarHistory     #
######################################


def test_min_scalar_history_equal_true() -> None:
    assert MinScalarHistory(
        "loss", elements=((None, 35), (1, 42), (2, 50)), best_value=50, improved=True
    ).equal(
        MinScalarHistory(
            "loss", elements=((None, 35), (1, 42), (2, 50)), best_value=50, improved=True
        )
    )


def test_min_scalar_history_equal_true_empty() -> None:
    assert MinScalarHistory("loss").equal(MinScalarHistory("loss"))


def test_min_scalar_history_equal_false_different_values() -> None:
    assert not MinScalarHistory(
        "loss", elements=((None, 35), (1, 42), (2, 50)), best_value=50, improved=True
    ).equal(
        MinScalarHistory(
            "loss", elements=((None, 35), (1, 42), (2, 51)), best_value=51, improved=True
        )
    )


def test_min_scalar_history_equal_false_different_names() -> None:
    assert not MinScalarHistory("loss").equal(MinScalarHistory("error"))


def test_min_scalar_history_equal_false_different_max_sizes() -> None:
    assert not MinScalarHistory("loss").equal(MinScalarHistory("loss", max_size=2))


def test_min_scalar_history_equal_false_different_types() -> None:
    assert not MinScalarHistory("loss").equal(MaxScalarHistory("loss"))


def test_min_scalar_get_best_value_last_is_best() -> None:
    history = MinScalarHistory("loss")
    history.add_value(2, step=0)
    history.add_value(1, step=1)
    assert history.get_best_value() == 1


def test_min_scalar_get_best_value_last_is_not_best() -> None:
    history = MinScalarHistory("loss")
    history.add_value(2, step=0)
    history.add_value(4, step=1)
    assert history.get_best_value() == 2


def test_min_scalar_to_dict_from_dict() -> None:
    assert BaseHistory.from_dict(
        MinScalarHistory(
            "loss", max_size=5, elements=[(0, 1)], best_value=1, improved=True
        ).to_dict()
    ).equal(MinScalarHistory("loss", max_size=5, elements=[(0, 1)], best_value=1, improved=True))
