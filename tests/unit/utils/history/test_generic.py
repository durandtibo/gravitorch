import torch
from objectory import OBJECT_TARGET
from pytest import mark, raises
from torch import Tensor

from gravitorch.utils.history import (
    BaseHistory,
    EmptyHistoryError,
    GenericHistory,
    MinScalarHistory,
    NotAComparableHistoryError,
)

####################################
#     Tests for GenericHistory     #
####################################


def test_generic_history_str_name():
    assert str(GenericHistory("loss")) == "GenericHistory(name=loss, max_size=10, history=())"


@mark.parametrize("name", ("name", "accuracy", ""))
def test_generic_history_init_name(name: str):
    assert GenericHistory(name).name == name


@mark.parametrize("max_size", (1, 5))
def test_generic_history_init_max_size(max_size: int):
    assert GenericHistory("loss", max_size=max_size).max_size == max_size


def test_generic_history_init_max_size_incorrect():
    with raises(ValueError):
        GenericHistory("loss", max_size=0)


def test_generic_history_add_value():
    assert GenericHistory("loss", elements=((None, "abc"), (1, 123))).get_recent_history() == (
        (None, "abc"),
        (1, 123),
    )


def test_generic_history_add_value_tensor():
    history = GenericHistory[Tensor]("predictions")
    history.add_value(torch.ones(2, 3))
    assert history.get_last_value().equal(torch.ones(2, 3))


def test_generic_history_clone():
    history = GenericHistory("loss", elements=((None, 35), (1, 42)))
    history_cloned = history.clone()
    assert history is not history_cloned
    assert history.equal(history_cloned)


def test_generic_history_clone_empty():
    history = GenericHistory("loss")
    history_cloned = history.clone()
    assert history is not history_cloned
    assert history.equal(history_cloned)


def test_generic_history_equal_true():
    assert GenericHistory("loss", elements=((None, 35), (1, 42))).equal(
        GenericHistory("loss", elements=((None, 35), (1, 42)))
    )


def test_generic_history_equal_true_empty():
    assert GenericHistory("loss").equal(GenericHistory("loss"))


def test_generic_history_equal_false_different_values():
    assert not GenericHistory("loss", elements=((None, 35), (1, 42))).equal(
        GenericHistory("loss", elements=((None, 35), (1, 50)))
    )


def test_generic_history_equal_false_different_names():
    assert not GenericHistory("loss", elements=((None, 35), (1, 42))).equal(
        GenericHistory("accuracy", elements=((None, 35), (1, 42)))
    )


def test_generic_history_equal_false_different_max_sizes():
    assert not GenericHistory("loss").equal(GenericHistory("loss", max_size=2))


def test_generic_history_equal_false_different_types():
    assert not GenericHistory("loss").equal(1)


def test_generic_history_equal_false_different_types_history():
    assert not GenericHistory("loss").equal(MinScalarHistory("loss"))


def test_generic_history_get_best_value():
    history = GenericHistory("loss")
    with raises(NotAComparableHistoryError):
        history.get_best_value()


def test_generic_history__get_best_value():
    history = GenericHistory("loss")
    with raises(NotImplementedError):
        history._get_best_value()


def test_generic_history_get_recent_history():
    assert GenericHistory("loss", elements=[(1, 123)]).get_recent_history() == ((1, 123),)


def test_generic_history_get_recent_history_empty():
    assert GenericHistory("loss").get_recent_history() == ()


def test_generic_history_get_recent_history_max_size():
    assert GenericHistory(
        "loss", max_size=3, elements=[(1, 123), (2, 123), (3, 124), (4, 125)]
    ).get_recent_history() == ((2, 123), (3, 124), (4, 125))


def test_generic_history_get_recent_history_add_max_size():
    history = GenericHistory("loss", max_size=3)
    for i in range(10):
        history.add_value(i)
    assert history.equal(
        GenericHistory("loss", max_size=3, elements=((None, 7), (None, 8), (None, 9)))
    )
    assert history.get_last_value() == 9


def test_generic_history_get_last_value():
    assert GenericHistory("loss", elements=((None, 35), (1, 42))).get_last_value() == 42


def test_generic_history_get_last_value_empty():
    history = GenericHistory("loss")
    with raises(EmptyHistoryError):
        history.get_last_value()


def test_generic_history_has_improved():
    history = GenericHistory("loss")
    with raises(NotAComparableHistoryError):
        history.has_improved()


def test_generic_history__has_improved():
    history = GenericHistory("loss")
    with raises(NotImplementedError):
        history._has_improved()


def test_generic_history_is_comparable():
    assert not GenericHistory("loss").is_comparable()


def test_generic_history_is_empty_true():
    assert GenericHistory("loss").is_empty()


def test_generic_history_is_empty_false():
    assert not GenericHistory("loss", elements=((None, 35), (1, 42))).is_empty()


def test_generic_history_config_dict():
    assert GenericHistory("loss").config_dict() == {
        OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
        "name": "loss",
        "max_size": 10,
    }


def test_generic_history_config_dict_create_new_history():
    assert BaseHistory.factory(
        **GenericHistory("loss", max_size=5, elements=[(0, 1)]).config_dict()
    ).equal(GenericHistory("loss", max_size=5))


def test_generic_history_load_state_dict_empty():
    history = GenericHistory("loss")
    history.load_state_dict({"history": ()})
    assert history.get_recent_history() == ()
    assert history.equal(GenericHistory("loss"))


def test_generic_history_load_state_dict_max_size_2():
    history = GenericHistory("loss", max_size=2)
    history.load_state_dict({"history": ((0, 1), (1, "abc"))})
    history.add_value(7, step=2)
    assert history.equal(GenericHistory("loss", max_size=2, elements=((1, "abc"), (2, 7))))


def test_generic_history_load_state_dict_reset():
    history = GenericHistory("loss", elements=[(0, 7)])
    history.load_state_dict({"history": ((0, 4), (1, 5))})
    assert history.equal(GenericHistory("loss", elements=((0, 4), (1, 5))))


def test_generic_history_state_dict():
    assert GenericHistory("loss", elements=((0, 1), (1, 5))).state_dict() == {
        "history": ((0, 1), (1, 5))
    }


def test_generic_history_state_dict_empty():
    assert GenericHistory("loss").state_dict() == {"history": ()}


def test_generic_history_to_dict():
    assert GenericHistory("loss", elements=[(0, 5)]).to_dict() == {
        "config": {
            OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
            "name": "loss",
            "max_size": 10,
        },
        "state": {"history": ((0, 5),)},
    }


def test_generic_history_to_dict_empty():
    assert GenericHistory("loss").to_dict() == {
        "config": {
            OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
            "name": "loss",
            "max_size": 10,
        },
        "state": {"history": ()},
    }


def test_generic_history_from_dict():
    assert BaseHistory.from_dict(
        {
            "config": {
                OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
                "name": "loss",
                "max_size": 7,
            },
            "state": {"history": ((0, 1), (1, 5))},
        }
    ).equal(GenericHistory("loss", max_size=7, elements=((0, 1), (1, 5))))


def test_generic_history_from_dict_empty():
    assert BaseHistory.from_dict(
        {
            "config": {
                OBJECT_TARGET: "gravitorch.utils.history.generic.GenericHistory",
                "name": "loss",
            },
            "state": {"history": ()},
        }
    ).equal(GenericHistory("loss"))
