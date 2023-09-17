from unittest.mock import Mock

import torch
from coola import objects_are_equal
from pytest import fixture
from torch import Tensor

from gravitorch.creators.datapipe import DictBatcherIterDataPipeCreator
from gravitorch.data.datacreators import BaseDataCreator, DataCreator
from gravitorch.datapipes.iter import DictBatcher
from gravitorch.engines import BaseEngine


@fixture
def datacreator() -> BaseDataCreator[dict[str, Tensor]]:
    return DataCreator({"key1": torch.ones(6, 3), "key2": torch.zeros(6)})


####################################################
#     Tests for DictBatcherIterDataPipeCreator     #
####################################################


def test_dict_batcher_iter_datapipe_creator_str(
    datacreator: BaseDataCreator[dict[str, Tensor]]
) -> None:
    assert str(DictBatcherIterDataPipeCreator(datacreator)).startswith(
        "DictBatcherIterDataPipeCreator("
    )


def test_dict_batcher_iter_datapipe_creator_create(
    datacreator: BaseDataCreator[dict[str, Tensor]]
) -> None:
    datapipe = DictBatcherIterDataPipeCreator(datacreator, batch_size=4).create()
    assert isinstance(datapipe, DictBatcher)
    assert objects_are_equal(
        tuple(datapipe),
        (
            {"key1": torch.ones(4, 3), "key2": torch.zeros(4)},
            {"key1": torch.ones(2, 3), "key2": torch.zeros(2)},
        ),
    )


def test_dict_batcher_iter_datapipe_creator_create_with_engine() -> None:
    engine = Mock(spec=BaseEngine)
    datacreator = Mock(
        spec=BaseDataCreator,
        create=Mock(return_value={"key1": torch.ones(6, 3), "key2": torch.zeros(6)}),
    )
    datapipe = DictBatcherIterDataPipeCreator(datacreator, batch_size=4).create(engine)
    datacreator.create.assert_called_once_with(engine)
    assert isinstance(datapipe, DictBatcher)
    assert objects_are_equal(
        tuple(datapipe),
        (
            {"key1": torch.ones(4, 3), "key2": torch.zeros(4)},
            {"key1": torch.ones(2, 3), "key2": torch.zeros(2)},
        ),
    )
