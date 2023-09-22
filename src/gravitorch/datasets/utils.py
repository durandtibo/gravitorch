from __future__ import annotations

__all__ = ["log_box_dataset_class"]

import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def log_box_dataset_class(dataset: Dataset) -> None:
    r"""Logs the name of the dataset in a "box".

    Args:
    ----
        dataset (``torch.utils.data.Dataset``): The dataset to log.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import DummyMultiClassDataset, log_box_dataset_class
        >>> dataset = DummyMultiClassDataset(num_examples=10, num_classes=5, feature_size=6)
        >>> log_box_dataset_class(dataset)
    """
    dataset_name = dataset.__class__.__qualname__
    logger.info(" " + "-" * (len(dataset_name) + 2))
    logger.info("| " + dataset_name + " |")
    logger.info(" " + "-" * (len(dataset_name) + 2))
