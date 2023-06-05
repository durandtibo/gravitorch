from pathlib import Path
from unittest.mock import patch

import torch
from pytest import TempPathFactory, fixture, raises

from gravitorch import constants as ct
from gravitorch.data.datasets import MNIST
from gravitorch.data.datasets.mnist import get_default_transform
from gravitorch.testing import torchvision_available
from gravitorch.utils.imports import is_torchvision_available
from gravitorch.utils.io import save_pytorch

if is_torchvision_available():
    from torchvision import transforms


@fixture(scope="module")
def mnist_path(tmp_path_factory: TempPathFactory) -> Path:
    num_examples = 5
    mock_data = (
        torch.zeros(num_examples, 28, 28, dtype=torch.uint8),
        torch.zeros(num_examples, dtype=torch.long),
    )
    path = tmp_path_factory.mktemp("data")
    save_pytorch(mock_data, path.joinpath("MNIST/processed/training.pt"))
    save_pytorch(mock_data, path.joinpath("MNIST/processed/test.pt"))
    return path


##################################
#     Tests for MNISTDataset     #
##################################


@torchvision_available
def test_mnist_dataset_getitem(mnist_path: Path) -> None:
    dataset = MNIST(mnist_path, transform=transforms.ToTensor())
    example = dataset[0]
    assert isinstance(example, dict)
    assert example[ct.INPUT].shape == (1, 28, 28)
    assert example[ct.INPUT].dtype == torch.float
    assert isinstance(example[ct.TARGET], int)


def test_mnist_dataset_without_torchvision(mnist_path: Path) -> None:
    with patch("gravitorch.utils.imports.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError, match="`torchvision` package is required but not installed."):
            MNIST(mnist_path)


@torchvision_available
def test_mnist_dataset_create_with_default_transforms(mnist_path: Path) -> None:
    dataset = MNIST.create_with_default_transforms(mnist_path)
    example = dataset[0]
    assert isinstance(example, dict)
    assert example[ct.INPUT].shape == (1, 28, 28)
    assert example[ct.INPUT].dtype == torch.float
    assert isinstance(example[ct.TARGET], int)


###########################################
#     Tests for get_default_transform     #
###########################################


@torchvision_available
def test_get_default_transform() -> None:
    transform = get_default_transform()
    assert isinstance(transform, transforms.Compose)
    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], transforms.ToTensor)
    assert isinstance(transform.transforms[1], transforms.Normalize)
