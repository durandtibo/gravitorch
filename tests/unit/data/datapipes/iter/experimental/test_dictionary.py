from unittest.mock import Mock

from pytest import raises

from gravitorch.data.datapipes.iter import SourceIterDataPipe
from gravitorch.data.datapipes.iter.experimental import (
    RenameAllKeysIterDataPipe,
    UpdateDictIterDataPipe,
)

###############################################
#     Tests for RenameAllKeysIterDataPipe     #
###############################################


def test_rename_all_keys_iter_datapipe_str():
    assert str(RenameAllKeysIterDataPipe(SourceIterDataPipe([]), key_mapping={})).startswith(
        "RenameAllKeysIterDataPipe("
    )


def test_rename_all_keys_iter_datapipe_iter():
    source = SourceIterDataPipe({"key1": 10 + i, "key2": 10 - i} for i in range(3))
    datapipe = RenameAllKeysIterDataPipe(
        source, key_mapping={"key1": "new_key1", "key2": "new_key2"}
    )
    assert list(datapipe) == [
        {"new_key1": 10, "new_key2": 10},
        {"new_key1": 11, "new_key2": 9},
        {"new_key1": 12, "new_key2": 8},
    ]


def test_rename_all_keys_iter_datapipe_len():
    datapipe = Mock(__len__=Mock(return_value=5))
    assert (
        len(
            RenameAllKeysIterDataPipe(
                datapipe, key_mapping={"key1": "new_key1", "key2": "new_key2"}
            )
        )
        == 5
    )


def test_rename_all_keys_iter_datapipe_no_len():
    source = SourceIterDataPipe({"key1": 10 + i, "key2": 10 - i} for i in range(3))
    with raises(TypeError):
        len(RenameAllKeysIterDataPipe(source, key_mapping={"key1": "new_key1", "key2": "new_key2"}))


############################################
#     Tests for UpdateDictIterDataPipe     #
############################################


def test_update_dict_iter_datapipe_str():
    assert str(UpdateDictIterDataPipe(SourceIterDataPipe([]), {})).startswith(
        "UpdateDictIterDataPipe("
    )


def test_update_dict_iter_datapipe_iter():
    assert tuple(
        UpdateDictIterDataPipe(
            SourceIterDataPipe([{"key": 42}, {"key": 43}, {"key": 44}]), {"other": 1}
        )
    ) == tuple([{"key": 42, "other": 1}, {"key": 43, "other": 1}, {"key": 44, "other": 1}])


def test_update_dict_iter_datapipe_iter_empty_dict():
    assert tuple(
        UpdateDictIterDataPipe(SourceIterDataPipe([{"key": 42}, {"key": 43}, {"key": 44}]), {})
    ) == tuple([{"key": 42}, {"key": 43}, {"key": 44}])


def test_update_dict_iter_datapipe_iter_duplicate_key():
    assert tuple(
        UpdateDictIterDataPipe(
            SourceIterDataPipe([{"key": 42}, {"key": 43}, {"key": 44}]), {"key": 1}
        )
    ) == tuple([{"key": 1}, {"key": 1}, {"key": 1}])


def test_update_dict_iter_datapipe_iter_empty():
    assert tuple(UpdateDictIterDataPipe(SourceIterDataPipe([]), {"key1": 1})) == tuple()


def test_update_dict_iter_datapipe_len():
    assert len(UpdateDictIterDataPipe(Mock(__len__=Mock(return_value=5)), {})) == 5


def test_update_dict_iter_datapipe_no_len():
    with raises(TypeError):
        len(UpdateDictIterDataPipe(Mock(), {}))
