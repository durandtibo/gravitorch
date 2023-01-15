from pathlib import Path
from unittest.mock import patch

import torch
from pytest import TempPathFactory, fixture, raises

from gravitorch import constants as ct
from gravitorch.data.datasets import MNISTDataset
from gravitorch.utils.integrations import is_torchvision_available
from gravitorch.utils.io import save_pytorch
from tests.testing import torchvision_available

if is_torchvision_available():
    from torchvision.transforms import ToTensor


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


##################################
#     Tests for MNISTDataset     #
##################################


@torchvision_available
def test_mnist_dataset_getitem(mnist_path: Path):
    dataset = MNISTDataset(root=mnist_path.as_posix(), transform=ToTensor())
    example = dataset[0]
    assert isinstance(example, dict)
    assert example[ct.INPUT].shape == (1, 28, 28)
    assert example[ct.INPUT].dtype == torch.float
    assert isinstance(example[ct.TARGET], int)


def test_mnist_dataset_without_torchvision():
    with patch("gravitorch.utils.integrations.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError):
            MNISTDataset()
