from gravitorch.utils import to_list, to_tuple

##############################
#     Tests for to_list     #
##############################


def test_to_list_list() -> None:
    assert to_list([1, 2, 3]) == [1, 2, 3]


def test_to_list_tuple() -> None:
    assert to_list((1, 2, 3)) == [1, 2, 3]


def test_to_list_bool() -> None:
    assert to_list(True) == [True]


def test_to_list_int() -> None:
    assert to_list(1) == [1]


def test_to_list_float() -> None:
    assert to_list(42.1) == [42.1]


def test_to_list_str() -> None:
    assert to_list("abc") == ["abc"]


##############################
#     Tests for to_tuple     #
##############################


def test_to_tuple_tuple() -> None:
    assert to_tuple((1, 2, 3)) == (1, 2, 3)


def test_to_tuple_list() -> None:
    assert to_tuple([1, 2, 3]) == (1, 2, 3)


def test_to_tuple_bool() -> None:
    assert to_tuple(True) == (True,)


def test_to_tuple_int() -> None:
    assert to_tuple(1) == (1,)


def test_to_tuple_float() -> None:
    assert to_tuple(42.1) == (42.1,)


def test_to_tuple_str() -> None:
    assert to_tuple("abc") == ("abc",)
