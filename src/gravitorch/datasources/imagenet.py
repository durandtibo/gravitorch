__all__ = ["ImageNetDataSource", "create_train_eval_datasets_v1"]

import logging
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import Dataset

from gravitorch import constants as ct
from gravitorch.creators.dataloader.base import BaseDataLoaderCreator
from gravitorch.data.datasets import ImageFolderDataset
from gravitorch.datasources.dataset import DatasetDataSource
from gravitorch.utils.integrations import check_torchvision, is_torchvision_available

if is_torchvision_available():
    from torchvision import transforms
else:
    transforms = None  # pragma: no cover

logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ImageNetDataSource(DatasetDataSource):
    r"""Implements a data source for the ImageNet dataset."""

    @classmethod
    def create_imagenet_v1(
        cls,
        train_path: Union[Path, str, None],
        eval_path: Union[Path, str, None],
        data_loader_creators: dict[str, Union[BaseDataLoaderCreator, dict, None]],
        input_size: int = 224,
    ) -> "ImageNetDataSource":
        r"""Creates a data source for the ImageNet dataset the training and
        evaluation datasets with their original data augmentations (V1).

        This function initializes the original data augmentations,
        sometimes called V1
        (https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/).
        Note that more aggressive data augmentation can be used to
        improve the performances. See the following article for more
        information:

            Wightman R., Touvron H., Jégou H.
            ResNet strikes back: An improved training procedure in timm.
            http://arxiv.org/pdf/2110.00476

        Args:
            train_path (``pathlib.Path`` or str or ``None``):
                Specifies the path to the training dataset.
                If ``None``, the training dataset is set to ``None``.
            eval_path (``pathlib.Path`` or str or ``None``):
                Specifies the path to the evaluation dataset.
                If ``None``, the evaluation dataset is set to ``None``.
            data_loader_creators: (dict): Specifies the data loader
                creators to initialize. Each key indicates a data
                loader creator name. For example if you want to create
                a data loader for ``'train'`` ID, the dictionary has to
                have a key ``'train'``. The value can be a
                ``BaseDataLoaderCreator`` object, or its configuration,
                or ``None``. ``None`` means a default data loader will
                be created. Each data loader creator takes a
                ``Dataset`` object as input, so you need to specify
                the associated dataset path.
            input_size (int, optional): Specifies the input size for
                the image. Default: ``224``

        Returns:
            ``ImageNetDataSource``: A data source for the ImageNet datasets.
        """
        train_dataset, eval_dataset = create_train_eval_datasets_v1(
            train_path=train_path,
            eval_path=eval_path,
            input_size=input_size,
        )
        datasets = {}
        if train_dataset:
            datasets[ct.TRAIN] = train_dataset
        if eval_dataset:
            datasets[ct.EVAL] = eval_dataset
        return cls(datasets=datasets, data_loader_creators=data_loader_creators)


def create_train_eval_datasets_v1(
    train_path: Union[Path, str, None],
    eval_path: Union[Path, str, None],
    input_size: int = 224,
) -> tuple[Optional[Dataset], Optional[Dataset]]:
    r"""Creates the training and evaluation datasets with the original data
    augmentations (V1).

    This function initializes the original data augmentations,
    sometimes called V1
    (https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/).
    Note that more aggressive data augmentation can be used to
    improve the performances. See the following article for more
    information:

        Wightman R., Touvron H., Jégou H.
        ResNet strikes back: An improved training procedure in timm.
        http://arxiv.org/pdf/2110.00476

    Args:
        train_path (``pathlib.Path`` or str or ``None``): Specifies
            the path to the training dataset. If ``None``, the
            training dataset is set to ``None``.
        eval_path (``pathlib.Path`` or str or ``None``): Specifies
            the path to the evaluation dataset. If ``None``, the
            evaluation dataset is set to ``None``.
        input_size (int, optional): Specifies the input size for the
            image. Default: ``224``

    Returns:
        (``Dataset`` or ``None``, ``Dataset`` or ``None``):
            The training and evaluation datasets. The datasets use the
            original image data augmentation (V1).
    """
    check_torchvision()
    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    train_dataset, eval_dataset = None, None
    if train_path:
        logger.info(f"Initialize the training dataset from {train_path}")
        train_dataset = ImageFolderDataset(
            train_path,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        logger.info(f"number of training examples: {len(train_dataset):,}")

    if eval_path:
        logger.info(f"Initialize the evaluation dataset from {eval_path}")
        eval_dataset = ImageFolderDataset(
            eval_path,
            transforms.Compose(
                [
                    # Maintain same ratio w.r.t. 256 -> 224 images
                    transforms.Resize(int((256 / 224) * input_size)),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        logger.info(f"number of evaluation examples: {len(eval_dataset):,}")
    return train_dataset, eval_dataset
