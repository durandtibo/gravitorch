from __future__ import annotations

from objectory import OBJECT_TARGET
from torch.utils.data.datapipes.iter import IterableWrapper

from gravitorch.creators.datapipe import DataPipeCreator

#####################################
#     Tests for DataPipeCreator     #
#####################################


def test_datapipe_creator_str() -> None:
    assert str(DataPipeCreator(IterableWrapper([1, 2, 3, 4, 5]))).startswith("DataPipeCreator(")


def test_datapipe_creator_create_datapipe_object() -> None:
    datapipe = IterableWrapper([1, 2, 3, 4, 5])
    dp = DataPipeCreator(datapipe).create()
    assert dp is datapipe
    assert tuple(dp) == tuple(datapipe)


def test_datapipe_creator_create_datapipe_config() -> None:
    assert tuple(
        DataPipeCreator(
            {
                OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
                "iterable": [1, 2, 3, 4, 5],
            }
        ).create()
    ) == tuple([1, 2, 3, 4, 5])


def test_datapipe_creator_create_deepcopy() -> None:
    datapipe = IterableWrapper([1, 2, 3, 4, 5])
    dp = DataPipeCreator(datapipe, deepcopy=True).create()
    assert dp is not datapipe
    assert tuple(dp) == tuple(datapipe)


def test_datapipe_creator_create_cache() -> None:
    creator = DataPipeCreator(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4, 5],
        },
        cache=True,
    )
    datapipe1 = creator.create()
    datapipe2 = creator.create()
    assert datapipe1 is datapipe2
    assert tuple(datapipe1) == tuple(datapipe2)


def test_datapipe_creator_create_no_cache() -> None:
    creator = DataPipeCreator(
        {
            OBJECT_TARGET: "torch.utils.data.datapipes.iter.IterableWrapper",
            "iterable": [1, 2, 3, 4, 5],
        },
    )
    datapipe1 = creator.create()
    datapipe2 = creator.create()
    assert datapipe1 is not datapipe2
    assert tuple(datapipe1) == tuple(datapipe2)
