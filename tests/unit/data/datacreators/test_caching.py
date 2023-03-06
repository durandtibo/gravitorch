from unittest.mock import Mock, patch

import torch
from pytest import mark

from gravitorch.data.datacreators import BaseDataCreator, OneCacheDataCreator
from gravitorch.engines import BaseEngine

#########################################
#     Tests for OneCacheDataCreator     #
#########################################


def test_one_cache_data_creator_str() -> None:
    assert str(OneCacheDataCreator(Mock(spec=BaseDataCreator))).startswith("OneCacheDataCreator(")


def test_one_cache_data_creator_data_creator() -> None:
    creator = Mock(spec=BaseDataCreator)
    assert OneCacheDataCreator(creator).data_creator is creator


@mark.parametrize("deepcopy", (True, False))
def test_one_cache_data_creator_deepcopy(deepcopy: bool) -> None:
    assert OneCacheDataCreator(Mock(spec=BaseDataCreator), deepcopy=deepcopy).deepcopy == deepcopy


def test_one_cache_data_creator_create_no_engine() -> None:
    creator = Mock(spec=BaseDataCreator, create=Mock(return_value=torch.ones(2, 3)))
    data_creator = OneCacheDataCreator(creator)
    data_creator.create().equal(torch.ones(2, 3))
    assert data_creator._is_cache_created
    assert data_creator._cached_data.equal(torch.ones(2, 3))
    creator.create.assert_called_once_with(None)


def test_one_cache_data_creator_create_engine() -> None:
    engine = Mock(spec=BaseEngine)
    creator = Mock(spec=BaseDataCreator, create=Mock(return_value=torch.ones(2, 3)))
    data_creator = OneCacheDataCreator(creator)
    data_creator.create(engine).equal(torch.ones(2, 3))
    assert data_creator._is_cache_created
    assert data_creator._cached_data.equal(torch.ones(2, 3))
    creator.create.assert_called_once_with(engine)


def test_one_cache_data_creator_create_repeat_deepcopy_false() -> None:
    creator = Mock(spec=BaseDataCreator, create=Mock(return_value=torch.ones(2, 3)))
    data_creator = OneCacheDataCreator(creator)
    data1 = data_creator.create()
    data2 = data_creator.create()
    assert data1 is data2
    assert data1.equal(data2)
    creator.create.assert_called_once_with(None)


def test_one_cache_data_creator_create_repeat_deepcopy_true() -> None:
    creator = Mock(spec=BaseDataCreator, create=Mock(return_value=torch.ones(2, 3)))
    data_creator = OneCacheDataCreator(creator, deepcopy=True)
    data1 = data_creator.create()
    data2 = data_creator.create()
    assert data1 is not data2
    assert data1.equal(data2)
    creator.create.assert_called_once_with(None)


def test_one_cache_data_creator_create_repeat_deepcopy_true_incorrect_type() -> None:
    creator = Mock(spec=BaseDataCreator, create=Mock(return_value=torch.ones(2, 3)))
    data_creator = OneCacheDataCreator(creator, deepcopy=True)
    with patch("gravitorch.data.datacreators.caching.copy.deepcopy", Mock(side_effect=TypeError)):
        data1 = data_creator.create()
        data2 = data_creator.create()
        assert data1 is data2
        assert data1.equal(data2)
        creator.create.assert_called_once_with(None)
