from unittest.mock import Mock

from pytest import raises

from gravitorch.data.datapipes.iter import SourceIterDataPipe
from gravitorch.data.datapipes.iter.experimental.sorting import (
    MultiKeysSeqMapSorterIterDataPipe,
)

#######################################################
#     Tests for MultiKeysSeqMapSorterIterDataPipe     #
#######################################################


def test_multi_keys_seq_map_sorter_iter_datapipe_str():
    assert str(MultiKeysSeqMapSorterIterDataPipe(SourceIterDataPipe([]), keys=[])).startswith(
        "MultiKeysSeqMapSorterIterDataPipe("
    )


def test_multi_keys_seq_map_sorter_iter_datapipe_iter_1_key():
    assert list(
        MultiKeysSeqMapSorterIterDataPipe(
            SourceIterDataPipe(
                [
                    [
                        {"a": 7, "b": 6, "c": 5},
                        {"a": 1, "b": 7, "c": 3},
                        {"a": 2, "b": 3, "c": 4},
                        {"a": 1, "b": 9, "c": 1},
                    ],
                    [
                        {"a": 7, "b": 6, "c": 5},
                        {"a": 8, "b": 3, "c": 1},
                        {"a": 2, "b": 3, "c": 4},
                    ],
                ]
            ),
            keys=["b"],
        )
    ) == [
        [
            {"a": 2, "b": 3, "c": 4},
            {"a": 7, "b": 6, "c": 5},
            {"a": 1, "b": 7, "c": 3},
            {"a": 1, "b": 9, "c": 1},
        ],
        [
            {"a": 8, "b": 3, "c": 1},
            {"a": 2, "b": 3, "c": 4},
            {"a": 7, "b": 6, "c": 5},
        ],
    ]


def test_multi_keys_seq_map_sorter_iter_datapipe_iter_2_keys():
    assert list(
        MultiKeysSeqMapSorterIterDataPipe(
            SourceIterDataPipe(
                [
                    [
                        {"a": 7, "b": 6, "c": 5},
                        {"a": 1, "b": 7, "c": 3},
                        {"a": 2, "b": 3, "c": 4},
                        {"a": 1, "b": 9, "c": 1},
                    ],
                    [
                        {"a": 7, "b": 6, "c": 5},
                        {"a": 8, "b": 3, "c": 1},
                        {"a": 2, "b": 3, "c": 4},
                    ],
                ]
            ),
            keys=["a", "b"],
        )
    ) == [
        [
            {"a": 1, "b": 7, "c": 3},
            {"a": 1, "b": 9, "c": 1},
            {"a": 2, "b": 3, "c": 4},
            {"a": 7, "b": 6, "c": 5},
        ],
        [
            {"a": 2, "b": 3, "c": 4},
            {"a": 7, "b": 6, "c": 5},
            {"a": 8, "b": 3, "c": 1},
        ],
    ]


def test_multi_keys_seq_map_sorter_iter_datapipe_len():
    assert (
        len(MultiKeysSeqMapSorterIterDataPipe(Mock(__len__=Mock(return_value=5)), keys=["key"]))
        == 5
    )


def test_multi_keys_seq_map_sorter_iter_datapipe_no_len():
    with raises(TypeError):
        len(
            MultiKeysSeqMapSorterIterDataPipe(SourceIterDataPipe(i for i in range(5)), keys=["key"])
        )
