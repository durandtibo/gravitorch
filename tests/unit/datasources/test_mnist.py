from pathlib import Path
from unittest.mock import Mock

import torch
from pytest import TempPathFactory, fixture, raises
from torch.utils.data import DataLoader

from gravitorch import constants as ct
from gravitorch.creators.dataloader import VanillaDataLoaderCreator
from gravitorch.datasources.mnist import MnistDataSource
from gravitorch.utils.asset import AssetNotFoundError
from gravitorch.utils.io import save_pytorch
from tests.testing import torchvision_available


@fixture(scope="module")
def mnist_path(tmp_path_factory: TempPathFactory) -> Path:
    num_examples = 5
    mock_data = (
        torch.zeros(num_examples, 28, 28, dtype=torch.uint8),
        torch.zeros(num_examples, dtype=torch.long),
    )
    path = tmp_path_factory.mktemp("data")
    save_pytorch(mock_data, path.joinpath("MNISTDataset/processed/training.pt"))
    save_pytorch(mock_data, path.joinpath("MNISTDataset/processed/test.pt"))
    return path


#####################################
#     Tests for MnistDataSource     #
#####################################


@torchvision_available
def test_mnist_data_source_str(mnist_path: Path):
    assert str(
        MnistDataSource(path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None})
    ).startswith("MnistDataSource(")


@torchvision_available
def test_mnist_data_source_attach(mnist_path: Path):
    MnistDataSource(path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}).attach(
        engine=Mock()
    )


@torchvision_available
def test_mnist_data_source_get_data_loader_train(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    assert isinstance(data_source.get_data_loader(ct.TRAIN), DataLoader)


@torchvision_available
def test_mnist_data_source_get_data_loader_eval(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    assert isinstance(data_source.get_data_loader(ct.EVAL), DataLoader)


@torchvision_available
def test_mnist_data_source_get_data_loader_batch_size_16(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path,
        data_loader_creators={ct.TRAIN: VanillaDataLoaderCreator(batch_size=16), ct.EVAL: None},
    )
    loader = data_source.get_data_loader(ct.TRAIN)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 16


@torchvision_available
def test_mnist_data_source_get_asset(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    with raises(AssetNotFoundError):
        data_source.get_asset("something")


@torchvision_available
def test_mnist_data_source_has_asset(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    assert not data_source.has_asset("something")


@torchvision_available
def test_mnist_data_source_load_state_dict(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    data_source.load_state_dict({})


@torchvision_available
def test_mnist_data_source_state_dict(mnist_path: Path):
    data_source = MnistDataSource(
        path=mnist_path, data_loader_creators={ct.TRAIN: None, ct.EVAL: None}
    )
    assert data_source.state_dict() == {}
