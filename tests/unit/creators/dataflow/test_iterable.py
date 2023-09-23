from __future__ import annotations

from objectory import OBJECT_TARGET

from gravitorch.creators.dataflow import IterableDataFlowCreator
from gravitorch.dataflows import IterableDataFlow


def create_list() -> list:
    return [1, 2, 3, 4, 5]


#############################################
#     Tests for IterableDataFlowCreator     #
#############################################


def test_iterable_dataflow_creator_str_with_length() -> None:
    print(str(IterableDataFlowCreator((1, 2, 3, 4, 5))))
    assert (
        str(IterableDataFlowCreator((1, 2, 3, 4, 5)))
        == "IterableDataFlowCreator(cache=False, length=5)"
    )


def test_iterable_dataflow_creator_str_without_length() -> None:
    assert (
        str(IterableDataFlowCreator(i for i in range(5))) == "IterableDataFlowCreator(cache=False)"
    )


def test_iterable_dataflow_creator_create() -> None:
    dataflow = IterableDataFlowCreator((1, 2, 3, 4, 5)).create()
    assert isinstance(dataflow, IterableDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]


def test_iterable_dataflow_creator_create_cache() -> None:
    creator = IterableDataFlowCreator(
        {OBJECT_TARGET: "tests.unit.creators.dataflow.test_iterable.create_list"}, cache=True
    )
    assert isinstance(creator._iterable, dict)
    dataflow = creator.create()
    assert creator._iterable == [1, 2, 3, 4, 5]
    assert isinstance(dataflow, IterableDataFlow)
    assert list(dataflow) == [1, 2, 3, 4, 5]


def test_iterable_dataflow_creator_create_deepcopy() -> None:
    dataflow = IterableDataFlowCreator((1, 2, 3, 4, 5), deepcopy=True).create()
    assert isinstance(dataflow, IterableDataFlow)
    assert dataflow._deepcopy
    assert list(dataflow) == [1, 2, 3, 4, 5]
