from pytest import raises

from gravitorch.utils.mapping import (
    convert_to_dict_of_lists,
    convert_to_list_of_dicts,
    get_first_value,
    remove_keys_starting_with,
)

##############################################
#     Tests for convert_to_dict_of_lists     #
##############################################


def test_convert_to_dict_of_lists_empty_list() -> None:
    assert convert_to_dict_of_lists([]) == {}


def test_convert_to_dict_of_lists_empty_dict() -> None:
    assert convert_to_dict_of_lists([{}]) == {}


def test_convert_to_dict_of_lists() -> None:
    assert convert_to_dict_of_lists(
        [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
    ) == {
        "key1": [1, 2, 3],
        "key2": [10, 20, 30],
    }


##############################################
#     Tests for convert_to_list_of_dicts     #
##############################################


def test_convert_to_list_of_dicts_empty_dict() -> None:
    assert convert_to_list_of_dicts({}) == []


def test_convert_to_list_of_dicts_empty_list() -> None:
    assert convert_to_list_of_dicts({"key1": [], "key2": []}) == []


def test_convert_to_list_of_dicts() -> None:
    assert convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]}) == [
        {"key1": 1, "key2": 10},
        {"key1": 2, "key2": 20},
        {"key1": 3, "key2": 30},
    ]


###############################################
#     Tests for remove_keys_starting_with     #
###############################################


def test_remove_keys_starting_with_empty() -> None:
    assert remove_keys_starting_with({}, "key") == {}


def test_remove_keys_starting_with() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
    ) == {
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_starting_with_another_key() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key."
    ) == {
        "key": 1,
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


#####################################
#     Tests for get_first_value     #
#####################################


def test_get_first_value_empty() -> None:
    with raises(ValueError, match="First value cannot be returned because the mapping is empty"):
        get_first_value({})


def test_get_first_value() -> None:
    assert get_first_value({"key1": 1, "key2": 2}) == 1
