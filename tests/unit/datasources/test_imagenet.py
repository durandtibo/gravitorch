from pathlib import Path
from unittest.mock import patch

from pytest import TempPathFactory, fixture, raises

from gravitorch import constants as ct
from gravitorch.creators.dataloader import AutoDataLoaderCreator
from gravitorch.data.datasets import ImageFolderDataset
from gravitorch.datasources.imagenet import (
    ImageNetDataSource,
    create_train_eval_datasets_v1,
)
from gravitorch.testing import torchvision_available
from tests.unit.data.datasets.test_image_folder import create_image_folder


@fixture(scope="module")
def dataset_path(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    create_image_folder(path)
    return path


########################################
#     Tests for ImageNetDataSource     #
########################################


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_no_train_and_no_eval() -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=None, eval_path=None, data_loader_creators={}
    )
    assert data_source._datasets == {}
    assert data_source._data_loader_creators == {}


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_no_train_and_eval(dataset_path: Path) -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=None,
        eval_path=dataset_path,
        data_loader_creators={},
    )
    assert len(data_source._datasets) == 1
    assert isinstance(data_source._datasets[ct.EVAL], ImageFolderDataset)
    assert data_source._data_loader_creators == {}


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_train_and_no_eval(dataset_path: Path) -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=dataset_path,
        eval_path=None,
        data_loader_creators={},
    )
    assert len(data_source._datasets) == 1
    assert isinstance(data_source._datasets[ct.TRAIN], ImageFolderDataset)
    assert data_source._data_loader_creators == {}


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_train_and_eval(dataset_path: Path) -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=dataset_path,
        eval_path=dataset_path,
        data_loader_creators={},
    )
    assert len(data_source._datasets) == 2
    assert isinstance(data_source._datasets[ct.TRAIN], ImageFolderDataset)
    assert isinstance(data_source._datasets[ct.EVAL], ImageFolderDataset)
    assert data_source._data_loader_creators == {}


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_input_size_224(dataset_path: Path) -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=dataset_path,
        eval_path=dataset_path,
        data_loader_creators={},
        input_size=224,
    )
    train_dataset = data_source._datasets[ct.TRAIN]
    assert isinstance(train_dataset, ImageFolderDataset)
    assert train_dataset.transform.transforms[0].size == (224, 224)  # RandomResizedCrop
    assert train_dataset[0][ct.INPUT].shape == (3, 224, 224)

    eval_dataset = data_source._datasets[ct.EVAL]
    assert isinstance(eval_dataset, ImageFolderDataset)
    assert eval_dataset.transform.transforms[0].size == 256  # Resize
    assert eval_dataset.transform.transforms[1].size == (224, 224)  # CenterCrop
    assert eval_dataset[0][ct.INPUT].shape == (3, 224, 224)


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_input_size_320(dataset_path: Path) -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=dataset_path,
        eval_path=dataset_path,
        data_loader_creators={},
        input_size=320,
    )
    train_dataset = data_source._datasets[ct.TRAIN]
    assert isinstance(train_dataset, ImageFolderDataset)
    assert train_dataset.transform.transforms[0].size == (320, 320)  # RandomResizedCrop
    assert train_dataset[0][ct.INPUT].shape == (3, 320, 320)

    eval_dataset = data_source._datasets[ct.EVAL]
    assert isinstance(eval_dataset, ImageFolderDataset)
    assert eval_dataset.transform.transforms[0].size == 365  # Resize
    assert eval_dataset.transform.transforms[1].size == (320, 320)  # CenterCrop
    assert eval_dataset[0][ct.INPUT].shape == (3, 320, 320)


@torchvision_available
def test_imagenet_data_source_create_imagenet_v1_data_loader_creators() -> None:
    data_source = ImageNetDataSource.create_imagenet_v1(
        train_path=None,
        eval_path=None,
        data_loader_creators={ct.TRAIN: None, ct.EVAL: None},
    )
    assert len(data_source._data_loader_creators) == 2
    assert isinstance(data_source._data_loader_creators[ct.TRAIN], AutoDataLoaderCreator)
    assert isinstance(data_source._data_loader_creators[ct.EVAL], AutoDataLoaderCreator)


###################################################
#     Tests for create_train_eval_datasets_v1     #
###################################################


@torchvision_available
def test_create_train_eval_datasets_v1_no_train_and_no_eval() -> None:
    assert create_train_eval_datasets_v1(train_path=None, eval_path=None) == (None, None)


@torchvision_available
def test_create_train_eval_datasets_v1_no_train_and_eval(dataset_path: Path) -> None:
    train_dataset, eval_dataset = create_train_eval_datasets_v1(
        train_path=None, eval_path=dataset_path
    )
    assert train_dataset is None
    assert isinstance(eval_dataset, ImageFolderDataset)


@torchvision_available
def test_create_train_eval_datasets_v1_train_and_no_eval(dataset_path: Path) -> None:
    train_dataset, eval_dataset = create_train_eval_datasets_v1(
        train_path=dataset_path, eval_path=None
    )
    assert isinstance(train_dataset, ImageFolderDataset)
    assert eval_dataset is None


@torchvision_available
def test_create_train_eval_datasets_v1_train_and_eval(dataset_path: Path) -> None:
    train_dataset, eval_dataset = create_train_eval_datasets_v1(
        train_path=dataset_path, eval_path=dataset_path
    )
    assert isinstance(train_dataset, ImageFolderDataset)
    assert isinstance(eval_dataset, ImageFolderDataset)


@torchvision_available
def test_create_train_eval_datasets_v1_input_size_224(dataset_path: Path) -> None:
    train_dataset, eval_dataset = create_train_eval_datasets_v1(
        train_path=dataset_path,
        eval_path=dataset_path,
        input_size=224,
    )
    assert isinstance(train_dataset, ImageFolderDataset)
    assert train_dataset.transform.transforms[0].size == (224, 224)  # RandomResizedCrop
    assert train_dataset[0][ct.INPUT].shape == (3, 224, 224)

    assert isinstance(eval_dataset, ImageFolderDataset)
    assert eval_dataset.transform.transforms[0].size == 256  # Resize
    assert eval_dataset.transform.transforms[1].size == (224, 224)  # CenterCrop
    assert eval_dataset[0][ct.INPUT].shape == (3, 224, 224)


@torchvision_available
def test_create_train_eval_datasets_v1_input_size_320(dataset_path: Path) -> None:
    train_dataset, eval_dataset = create_train_eval_datasets_v1(
        train_path=dataset_path,
        eval_path=dataset_path,
        input_size=320,
    )
    assert isinstance(train_dataset, ImageFolderDataset)
    assert train_dataset.transform.transforms[0].size == (320, 320)  # RandomResizedCrop
    assert train_dataset[0][ct.INPUT].shape == (3, 320, 320)

    assert isinstance(eval_dataset, ImageFolderDataset)
    assert eval_dataset.transform.transforms[0].size == 365  # Resize
    assert eval_dataset.transform.transforms[1].size == (320, 320)  # CenterCrop
    assert eval_dataset[0][ct.INPUT].shape == (3, 320, 320)


def test_create_train_eval_datasets_v1_without_torchvision() -> None:
    with patch("gravitorch.utils.imports.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError, match="`torchvision` package is required but not installed."):
            create_train_eval_datasets_v1(train_path="", eval_path="", input_size=320)
