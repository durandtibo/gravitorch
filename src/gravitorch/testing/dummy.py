r"""The code is in this module is designed to be used for testing
purpose only."""

from __future__ import annotations

__all__ = [
    "DummyDataset",
    "DummyIterableDataset",
    "DummyClassificationModel",
    "DummyDataSource",
    "create_dummy_engine",
]

from collections.abc import Iterator

import torch
from objectory import OBJECT_TARGET
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, IterableDataset

from gravitorch import constants as ct
from gravitorch.creators.dataloader import DataLoaderCreator
from gravitorch.datasources.base import BaseDataSource
from gravitorch.datasources.dataset import DatasetDataSource
from gravitorch.engines.base import BaseEngine
from gravitorch.lr_schedulers.base import LRSchedulerType
from gravitorch.models.base import BaseModel


class DummyDataset(Dataset):
    r"""Implements a dummy map-style dataset for testing purpose.

    Args:
    ----
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_examples (dim, optional): Specifies the number of
            examples. Default: ``8``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import DummyDataset
        >>> dataset = DummyDataset(num_examples=10, feature_size=7)
        >>> dataset[0]
        {'input': tensor([1., 1., 1., 1., 1., 1., 1.]), 'target': 1}
    """

    def __init__(self, feature_size: int = 4, num_examples: int = 8) -> None:
        self._feature_size = int(feature_size)
        self._num_examples = int(num_examples)

    def __getitem__(self, item: int) -> dict:
        return {ct.INPUT: torch.ones(self._feature_size) + item, ct.TARGET: 1}

    def __len__(self) -> int:
        return self._num_examples

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_examples={self._feature_size:,}, "
            f"feature_size={self._feature_size:,})"
        )


class DummyIterableDataset(IterableDataset):
    r"""Implements a dummy iterable-style dataset for testing purpose.

    Args:
    ----
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_examples (dim, optional): Specifies the number of
            examples. Default: ``8``
        has_length (bool, optional): If ``True``, the length of the
            dataset is defined, otherwise it returns ``TypeError``.
            Default: ``False``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import DummyIterableDataset
        >>> dataset = DummyIterableDataset(num_examples=10, feature_size=7)
        >>> next(iter(dataset))
        {'input': tensor([2., 2., 2., 2., 2., 2., 2.]), 'target': 1}
    """

    def __init__(
        self, feature_size: int = 4, num_examples: int = 8, has_length: bool = False
    ) -> None:
        self._feature_size = int(feature_size)
        self._num_examples = int(num_examples)
        self._has_length = bool(has_length)
        self._iteration = 0

    def __iter__(self) -> Iterator:
        self._iteration = 0
        return self

    def __next__(self) -> dict:
        self._iteration += 1
        if self._iteration > self._num_examples:
            raise StopIteration

        return {ct.INPUT: torch.ones(self._feature_size) + self._iteration, ct.TARGET: 1}

    def __len__(self) -> int:
        if self._has_length:
            return self._num_examples
        raise TypeError(f"{type(self).__qualname__} instance doesn't have valid length")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_examples={self._feature_size:,}, "
            f"feature_size={self._feature_size:,})"
        )


class DummyDataSource(DatasetDataSource):
    r"""Implements a dummy datasource for testing purpose.

    Args:
    ----
        train_dataset (``Dataset`` or ``None``, optional): Specifies
            the training dataset. If ``None``, a dummy map-style
            dataset is automatically created. Default: ``None``
        eval_dataset (``Dataset`` or ``None``, optional): Specifies
            the evaluation dataset. If ``None``, a dummy map-style
            dataset is automatically created. Default: ``None``
        batch_size (int, optional): Specifies the batch size.
            Default: ``1``

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import DummyDataset, DummyDataSource
        >>> datasource = DummyDataSource(
        ...     train_dataset=DummyDataset(num_examples=10, feature_size=3),
        ...     eval_dataset=DummyDataset(num_examples=6, feature_size=3),
        ... )
        >>> next(iter(datasource.get_iterable("train")))
        {'input': tensor([[1., 1., 1.]]), 'target': tensor([1])}
        >>> next(iter(datasource.get_iterable("eval")))
        {'input': tensor([[1., 1., 1.]]), 'target': tensor([1])}
    """

    def __init__(
        self,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        batch_size: int | None = 1,
    ) -> None:
        if train_dataset is None:
            train_dataset = DummyDataset()
        if eval_dataset is None:
            eval_dataset = DummyDataset()
        super().__init__(
            datasets={ct.TRAIN: train_dataset, ct.EVAL: eval_dataset},
            dataloader_creators={
                ct.TRAIN: DataLoaderCreator(batch_size=batch_size, shuffle=False),
                ct.EVAL: DataLoaderCreator(batch_size=batch_size, shuffle=False),
            },
        )


class DummyClassificationModel(BaseModel):
    r"""Implements a dummy classification model for testing purpose.

    Args:
    ----
        feature_size (dim, optional): Specifies the feature size.
            Default: ``4``
        num_classes (dim, optional): Specifies the number of classes.
            Default: ``3``
        loss_nan (bool, optional): If ``True``, the forward function
            returns a loss filled with a NaN value.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import DummyClassificationModel
        >>> model = DummyClassificationModel()
        >>> model
        DummyClassificationModel(
          (linear): Linear(in_features=4, out_features=3, bias=True)
          (criterion): CrossEntropyLoss()
        )
        >>> model({"input": torch.ones(2, 4), "target": torch.ones(2, dtype=torch.long)})
        {'loss': tensor(..., grad_fn=<NllLossBackward0>)}
    """

    def __init__(self, feature_size: int = 4, num_classes: int = 3, loss_nan: bool = False) -> None:
        super().__init__()
        self.linear = Linear(feature_size, num_classes)
        self.criterion = CrossEntropyLoss()
        self._return_loss_nan = bool(loss_nan)

    def forward(self, batch: dict) -> dict:
        if self._return_loss_nan:
            return {ct.LOSS: torch.tensor(float("nan"))}
        return {ct.LOSS: self.criterion(self.linear(batch[ct.INPUT]), batch[ct.TARGET])}


def create_dummy_engine(
    datasource: BaseDataSource | dict | None = None,
    model: Module | dict | None = None,
    optimizer: Optimizer | dict | None = None,
    lr_scheduler: LRSchedulerType | dict | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> BaseEngine:
    r"""Creates an engine with dummy components for testing purpose.

    Args:
    ----
        datasource (``BaseDataSource`` or dict or ``None``): Specifies
            the datasource or its configuration. If ``None``, a dummy
            datasource is automatically created. Default: ``None``
        model (``Module`` or dict or ``None``): Specifies the model or
            its configuration. If ``None``, a dummy classification model
            is automatically created. Default: ``None``
        optimizer (``Optimizer`` or dict or ``None``): Specifies the
            optimizer or its configuration. If ``None``, a SGD
            optimizer is automatically created. Default: ``None``
        device (``torch.device`` or ``None``): Specifies the target
            device. Default: ``None``
        **kwargs: Arbitrary keyword arguments.

    Returns:
    -------
        ``BaseEngine``: The initialized engine.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.testing import create_dummy_engine
        >>> engine = create_dummy_engine()
    """
    datasource = datasource or DummyDataSource(batch_size=2)
    model = model or DummyClassificationModel()
    optimizer = optimizer or {OBJECT_TARGET: "torch.optim.SGD", "lr": 0.01}

    # Local imports to avoid circular imports
    from gravitorch.creators.core import CoreCreator
    from gravitorch.engines import AlphaEngine

    return AlphaEngine(
        core_creator=CoreCreator(
            datasource=datasource,
            model=model.to(device=device),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        ),
        **kwargs,
    )
