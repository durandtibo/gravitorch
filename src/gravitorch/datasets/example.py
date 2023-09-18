r"""This module implements datasets that stores all the examples in
memory."""
from __future__ import annotations

__all__ = ["ExampleDataset", "ExampleDatasetEqualityOperator"]

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from coola import (
    BaseEqualityOperator,
    BaseEqualityTester,
    EqualityTester,
    objects_are_equal,
)
from torch.utils.data import Dataset

from gravitorch.datasets.utils import log_box_dataset_class
from gravitorch.utils.io import load_json, load_pickle
from gravitorch.utils.path import sanitize_path

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExampleDataset(Dataset[T]):
    r"""Implements a dataset that stores all the examples in-memory.

    You can use this dataset only if all the examples can fit
    in-memory.

    Args:
    ----
        examples: Specifies the examples of the dataset.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.datasets import ExampleDataset
        >>> dataset = ExampleDataset((1, 2, 3, 4, 5, 6))
        >>> dataset
        ExampleDataset(num_examples=6)
        >>> dataset[0]
        1
    """

    def __init__(self, examples: Sequence[T]) -> None:
        log_box_dataset_class(self)
        self._examples = tuple(examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, item: int) -> T:
        return self._examples[item]

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(num_examples={len(self):,})"

    def equal(self, other: Any) -> bool:
        r"""Indicates if two datasets are equal or not.

        Args:
        ----
            other: Specifies the other dataset to compare.

        Returns:
        -------
            bool: ``True`` if the datasets are equal,
                otherwise ``False``

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datasets import ExampleDataset
            >>> ExampleDataset([1, 2]).equal(ExampleDataset([1, 2]))
            True
            >>> ExampleDataset([1, 2]).equal(ExampleDataset([2, 1]))
            False
        """
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return objects_are_equal(self._examples, other._examples)

    @classmethod
    def from_json_file(cls, path: Path | str) -> ExampleDataset:
        r"""Instantiates a dataset with the examples from a JSON file.

        Args:
        ----
            path (``pathlib.Path`` or str): Specifies the path to the
                JSON file.

        Returns:
        -------
            ``ExampleDataset``: An instantiated dataset.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datasets import ExampleDataset
            >>> dataset = ExampleDataset.from_json_file("/path/to/file.pt")  # doctest: +SKIP
        """
        return cls(load_json(sanitize_path(path)))

    @classmethod
    def from_pickle_file(cls, path: Path | str) -> ExampleDataset:
        r"""Instantiates a dataset with the examples from a pickle file.

        Args:
        ----
            path (``pathlib.Path`` or str): Specifies the path to the
                pickle file.

        Returns:
        -------
            ``ExampleDataset``: An instantiated dataset.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datasets import ExampleDataset
            >>> dataset = ExampleDataset.from_pickle_file("/path/to/file.pkl")  # doctest: +SKIP
        """
        return cls(load_pickle(sanitize_path(path)))

    @classmethod
    def from_pytorch_file(cls, path: Path | str, **kwargs) -> ExampleDataset:
        r"""Instantiates a dataset with the examples from a PyTorch file.

        Args:
        ----
            path (``pathlib.Path`` or str): Specifies the path to the
                PyTorch file.
            **kwargs: See ``torch.load`` documentation.

        Returns:
        -------
            ``ExampleDataset``: An instantiated dataset.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.datasets import ExampleDataset
            >>> dataset = ExampleDataset.from_pytorch_file("/path/to/file.pt")  # doctest: +SKIP
        """
        return cls(torch.load(sanitize_path(path), **kwargs))


class ExampleDatasetEqualityOperator(BaseEqualityOperator[ExampleDataset]):
    r"""Implements an equality operator for ``ExampleDataset``
    objects."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> ExampleDatasetEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: ExampleDataset,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if not isinstance(object2, ExampleDataset):
            if show_difference:
                logger.info(f"object2 is not a `ExampleDataset` object: {type(object2)}")
            return False
        object_equal = object1.equal(object2)
        if show_difference and not object_equal:
            logger.info(
                f"`ExampleDataset` objects are different\nobject1=\n{object1}\nobject2=\n{object2}"
            )
        return object_equal


if not EqualityTester.has_operator(ExampleDataset):
    EqualityTester.add_operator(
        ExampleDataset, ExampleDatasetEqualityOperator()
    )  # pragma: no cover
