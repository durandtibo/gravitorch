from __future__ import annotations

from objectory import OBJECT_TARGET

from gravitorch.creators.datastream import IterableDataStreamCreator
from gravitorch.datastreams import IterableDataStream


def create_list() -> list:
    return [1, 2, 3, 4, 5]


#############################################
#     Tests for IterableDataStreamCreator     #
#############################################


def test_iterable_datastream_creator_str_with_length() -> None:
    print(str(IterableDataStreamCreator((1, 2, 3, 4, 5))))
    assert (
        str(IterableDataStreamCreator((1, 2, 3, 4, 5)))
        == "IterableDataStreamCreator(cache=False, length=5)"
    )


def test_iterable_datastream_creator_str_without_length() -> None:
    assert (
        str(IterableDataStreamCreator(i for i in range(5)))
        == "IterableDataStreamCreator(cache=False)"
    )


def test_iterable_datastream_creator_create() -> None:
    datastream = IterableDataStreamCreator((1, 2, 3, 4, 5)).create()
    assert isinstance(datastream, IterableDataStream)
    assert list(datastream) == [1, 2, 3, 4, 5]


def test_iterable_datastream_creator_create_cache() -> None:
    creator = IterableDataStreamCreator(
        {OBJECT_TARGET: "tests.unit.creators.datastream.test_iterable.create_list"}, cache=True
    )
    assert isinstance(creator._iterable, dict)
    datastream = creator.create()
    assert creator._iterable == [1, 2, 3, 4, 5]
    assert isinstance(datastream, IterableDataStream)
    assert list(datastream) == [1, 2, 3, 4, 5]


def test_iterable_datastream_creator_create_deepcopy() -> None:
    datastream = IterableDataStreamCreator((1, 2, 3, 4, 5), deepcopy=True).create()
    assert isinstance(datastream, IterableDataStream)
    assert datastream._deepcopy
    assert list(datastream) == [1, 2, 3, 4, 5]
