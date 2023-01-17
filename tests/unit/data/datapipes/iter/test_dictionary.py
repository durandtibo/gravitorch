from unittest.mock import Mock

from pytest import raises

from gravitorch.data.datapipes.iter import (
    DictOfListConverter,
    ListOfDictConverter,
    SourceIterDataPipe,
)

#########################################
#     Tests for DictOfListConverter     #
#########################################


def test_dict_of_list_converter_str():
    assert str(DictOfListConverter(SourceIterDataPipe([]))).startswith(
        "DictOfListConverterIterDataPipe("
    )


def test_dict_of_list_converter_iter():
    assert tuple(
        DictOfListConverter(
            SourceIterDataPipe(
                [
                    [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}],
                    [{"key": "a"}, {"key": -2}],
                ]
            )
        )
    ) == tuple(
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


def test_dict_of_list_converter_len():
    assert len(DictOfListConverter(Mock(__len__=Mock(return_value=5)))) == 5


def test_dict_of_list_converter_no_len():
    with raises(TypeError):
        len(DictOfListConverter(Mock()))


#########################################
#     Tests for ListOfDictConverter     #
#########################################


def test_list_of_dict_converter_str():
    assert str(ListOfDictConverter(SourceIterDataPipe([]))).startswith(
        "ListOfDictConverterIterDataPipe("
    )


def test_list_of_dict_converter_iter():
    assert tuple(
        ListOfDictConverter(
            SourceIterDataPipe(
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
    ) == tuple(
        [
            [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}],
            [{"key": "a"}, {"key": -2}],
        ]
    )


def test_list_of_dict_converter_len():
    assert len(ListOfDictConverter(Mock(__len__=Mock(return_value=5)))) == 5


def test_list_of_dict_converter_no_len():
    with raises(TypeError):
        len(ListOfDictConverter(Mock()))
