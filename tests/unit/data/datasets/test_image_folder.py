from pathlib import Path
from unittest.mock import patch

import numpy as np
from pytest import TempPathFactory, fixture, raises

from gravitorch import constants as ct
from gravitorch.data.datasets import ImageFolderDataset
from gravitorch.testing import pillow_available, torchvision_available
from gravitorch.utils.integrations import is_pillow_available

if is_pillow_available():
    from PIL import Image


def create_image_folder(path: Path) -> None:
    r"""Creates an image folder dataset with 2 classes: cat vs dog.

    Args:
    ----
        path (str): Specifies the path where to write the images of the dataset.
    """
    cat_path = path.joinpath("cat")
    cat_path.mkdir(exist_ok=True, parents=True)
    for n in range(3):
        im_out = Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")).convert("RGB")
        im_out.save(cat_path.joinpath(f"out{n}.jpg"))
    dog_path = path.joinpath("dog")
    dog_path.mkdir(exist_ok=True, parents=True)
    for n in range(2):
        im_out = Image.fromarray((np.random.rand(16, 16, 3) * 255).astype("uint8")).convert("RGB")
        im_out.save(dog_path.joinpath(f"out{n}.jpg"))


@fixture(scope="module")
def dataset_path(tmp_path_factory: TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    create_image_folder(path)
    return path


@torchvision_available
@pillow_available
def test_image_folder_dataset(dataset_path: Path) -> None:
    dataset = ImageFolderDataset(dataset_path.as_posix())
    assert len(dataset) == 5
    for i in range(3):
        assert len(dataset[i]) == 2
        assert ct.INPUT in dataset[i]
        assert dataset[i][ct.TARGET] == 0
    for i in range(3, 5):
        assert len(dataset[i]) == 2
        assert ct.INPUT in dataset[i]
        assert dataset[i][ct.TARGET] == 1


def test_image_folder_dataset_without_torchvision() -> None:
    with patch("gravitorch.utils.integrations.is_torchvision_available", lambda *args: False):
        with raises(RuntimeError):
            ImageFolderDataset()


def test_image_folder_dataset_without_pillow() -> None:
    with patch("gravitorch.utils.integrations.is_torchvision_available", lambda *args: True):
        with patch("gravitorch.utils.integrations.is_pillow_available", lambda *args: False):
            with raises(RuntimeError):
                ImageFolderDataset()
