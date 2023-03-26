from unittest.mock import Mock

from pytest import raises

from gravitorch.datapipes.iter import (
    DictOfListConverter,
    ListOfDictConverter,
    SourceWrapper,
)

#########################################
#     Tests for DictOfListConverter     #
#########################################


def test_dict_of_list_converter_str() -> None:
    assert str(DictOfListConverter(SourceWrapper([]))).startswith(
        "DictOfListConverterIterDataPipe("
    )


def test_dict_of_list_converter_iter() -> None:
    assert tuple(
        DictOfListConverter(
            SourceWrapper(
                [
                    [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}],
                    [{"key": "a"}, {"key": -2}],
                ]
            )
        )
    ) == (
        {
            "key1": [1, 2, 3],
            "key2": [10, 20, 30],
        },
        {
            "key": ["a", -2],
        },
    )


def test_dict_of_list_converter_len() -> None:
    assert len(DictOfListConverter(Mock(__len__=Mock(return_value=5)))) == 5


def test_dict_of_list_converter_no_len() -> None:
    with raises(TypeError):
        len(DictOfListConverter(Mock()))


#########################################
#     Tests for ListOfDictConverter     #
#########################################


def test_list_of_dict_converter_str() -> None:
    assert str(ListOfDictConverter(SourceWrapper([]))).startswith(
        "ListOfDictConverterIterDataPipe("
    )


def test_list_of_dict_converter_iter() -> None:
    assert tuple(
        ListOfDictConverter(
            SourceWrapper(
                [
                    {
                        "key1": [1, 2, 3],
                        "key2": [10, 20, 30],
                    },
                    {
                        "key": ["a", -2],
                    },
                ]
            )
        )
    ) == (
        [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}],
        [{"key": "a"}, {"key": -2}],
    )


def test_list_of_dict_converter_len() -> None:
    assert len(ListOfDictConverter(Mock(__len__=Mock(return_value=5)))) == 5


def test_list_of_dict_converter_no_len() -> None:
    with raises(TypeError):
        len(ListOfDictConverter(Mock()))
