from unittest.mock import Mock

from pytest import raises

from gravitorch.data.datapipes.iter import (
    SourceIterDataPipe,
    ToDictOfListIterDataPipe,
    ToListOfDictIterDataPipe,
)

##############################################
#     Tests for ToDictOfListIterDataPipe     #
##############################################


def test_to_dict_of_list_iter_datapipe_str():
    assert str(ToDictOfListIterDataPipe(SourceIterDataPipe([]))).startswith(
        "ToDictOfListIterDataPipe("
    )


def test_to_dict_of_list_iter_datapipe_iter():
    assert tuple(
        ToDictOfListIterDataPipe(
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


def test_to_dict_of_list_iter_datapipe_len():
    assert len(ToDictOfListIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_to_dict_of_list_iter_datapipe_no_len():
    with raises(TypeError):
        len(ToDictOfListIterDataPipe(Mock()))


##############################################
#     Tests for ToListOfDictIterDataPipe     #
##############################################


def test_to_list_of_dict_iter_datapipe_str():
    assert str(ToListOfDictIterDataPipe(SourceIterDataPipe([]))).startswith(
        "ToListOfDictIterDataPipe("
    )


def test_to_list_of_dict_iter_datapipe_iter():
    assert tuple(
        ToListOfDictIterDataPipe(
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


def test_to_list_of_dict_iter_datapipe_len():
    assert len(ToListOfDictIterDataPipe(Mock(__len__=Mock(return_value=5)))) == 5


def test_to_list_of_dict_iter_datapipe_no_len():
    with raises(TypeError):
        len(ToListOfDictIterDataPipe(Mock()))
