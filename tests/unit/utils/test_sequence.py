from gravitorch.utils import to_tuple

##############################
#     Tests for to_tuple     #
##############################


def test_to_tuples_tuple():
    assert to_tuple((1, 2, 3)) == (1, 2, 3)


def test_to_tuples_list():
    assert to_tuple([1, 2, 3]) == (1, 2, 3)


def test_to_tuples_bool():
    assert to_tuple(True) == (True,)


def test_to_tuples_int():
    assert to_tuple(1) == (1,)


def test_to_tuples_float():
    assert to_tuple(42.1) == (42.1,)


def test_to_tuples_str():
    assert to_tuple("abc") == ("abc",)
