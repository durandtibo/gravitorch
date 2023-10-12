import torch
from coola import objects_are_equal
from pytest import raises
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.datapipes.iter import Batcher, IterableWrapper

from gravitorch.datastreams import DataLoaderDataStream
from gravitorch.testing import torchdata_available
from gravitorch.utils.imports import is_torchdata_available

if is_torchdata_available():
    from torchdata.dataloader2 import DataLoader2

########################################
#     Tests for DataLoaderDataStream     #
########################################


def test_dataloader_datastream_str_with_length() -> None:
    assert str(DataLoaderDataStream(DataLoader(TensorDataset(torch.arange(10))))) == (
        "DataLoaderDataStream(length=10)"
    )


def test_dataloader_datastream_str_without_length() -> None:
    assert str(DataLoaderDataStream(DataLoader(i for i in range(5)))) == ("DataLoaderDataStream()")


def test_dataloader_datastream_incorrect_type() -> None:
    with raises(
        TypeError, match="Incorrect type. Expecting DataLoader or DataLoader2 but received"
    ):
        DataLoaderDataStream([1, 2, 3, 4, 5])


def test_dataloader_datastream_iter_dataloader() -> None:
    with DataLoaderDataStream(DataLoader(TensorDataset(torch.arange(10)), batch_size=4)) as flow:
        assert objects_are_equal(
            list(flow),
            [[torch.tensor([0, 1, 2, 3])], [torch.tensor([4, 5, 6, 7])], [torch.tensor([8, 9])]],
        )


@torchdata_available
def test_dataloader_datastream_iter_dataloader2() -> None:
    with DataLoaderDataStream(
        DataLoader2(Batcher(IterableWrapper(list(range(10))), batch_size=4))
    ) as flow:
        assert objects_are_equal(
            [list(batch) for batch in flow], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
        )


def test_dataloader_datastream_len() -> None:
    with DataLoaderDataStream(DataLoader(TensorDataset(torch.arange(10)), batch_size=4)) as flow:
        assert len(flow) == 3
