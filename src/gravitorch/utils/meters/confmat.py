from __future__ import annotations

__all__ = [
    "BaseConfusionMatrix",
    "BinaryConfusionMatrix",
    "MulticlassConfusionMatrix",
    "str_binary_confusion_matrix",
]

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from coola.utils import str_indent
from tabulate import tabulate
from torch import Tensor

from gravitorch.distributed.ddp import SUM, sync_reduce_
from gravitorch.utils.meters.exceptions import EmptyMeterError


class BaseConfusionMatrix:
    r"""Defines the base class to implement confusion matrix.

    Args:
    ----
        matrix (``torch.Tensor`` of type long and shape
            ``(num_classes, num_classes)``): Specifies the initial
            confusion matrix values. The rows indicate the true
            labels and the columns indicate the predicted labels.
    """

    def __init__(self, matrix: Tensor) -> None:
        check_confusion_matrix(matrix)
        self._matrix = matrix
        self._num_predictions = self._compute_num_predictions()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n"
            f"  num_predictions={self.num_predictions:,}\n"
            f"  shape={self._matrix.shape}\n"
            f"  dtype={self._matrix.dtype}\n"
            f"  {str_indent(self._matrix)}\n)"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(num_classes={self.num_classes:,}, "
            f"num_predictions={self.num_predictions:,})"
        )

    @property
    def matrix(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,
        num_classes)``: The confusion matrix values.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.matrix
            tensor([[2, 0],
                    [1, 3]])
        """
        return self._matrix

    @property
    def num_classes(self) -> int:
        r"""``int``: The number of classes.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.num_classes
            2
        """
        return self._matrix.shape[0]

    @property
    def num_predictions(self) -> int:
        r"""``int``: The number of predictions.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.num_predictions
            6
        """
        return self._num_predictions

    def all_reduce(self) -> None:
        r"""Reduces the meter values across all machines in such a way
        that all get the final result.

        The confusion matrix is reduced by summing all the confusion
        matrices (1 confusion matrix per distributed process).

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat_reduced = confmat.all_reduce()
        """
        sync_reduce_(self._matrix, SUM)
        # It is necessary to recompute the number of predictions because
        # the confusion matrix may have changed
        self._num_predictions = self._compute_num_predictions()

    def get_normalized_matrix(self, normalization: str) -> Tensor:
        r"""Gets the normalized confusion matrix.

        Args:
        ----
            normalization (str): Specifies the normalization strategy.
                The supported normalization strategies are:

                    - ``'true'``: normalization over the targets
                        (most commonly used)
                    - ``'pred'``: normalization over the predictions
                    - ``'all'``: normalization over the whole matrix

        Returns:
        -------
            ``torch.Tensor`` of type float and shape
                ``(num_classes, num_classes)``: The normalized
                confusion matrix.

        Raises:
        ------
            ValueError if the normalization strategy is not supported.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.get_normalized_matrix(normalization="true")
            tensor([[1.0000, 0.0000],
                    [0.2500, 0.7500]])
            >>> confmat.get_normalized_matrix(normalization="pred")
            tensor([[0.6667, 0.0000],
                    [0.3333, 1.0000]])
            >>> confmat.get_normalized_matrix(normalization="all")
            tensor([[0.3333, 0.0000],
                    [0.1667, 0.5000]])
        """
        if normalization == "true":
            # Clamp to avoid division by 0
            return self.matrix / self.matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        if normalization == "pred":
            # Clamp to avoid division by 0
            return self.matrix / self.matrix.sum(dim=0, keepdim=True).clamp(min=1e-8)
        if normalization == "all":
            # Clamp to avoid division by 0
            return self.matrix / self.matrix.sum().clamp(min=1e-8)
        raise ValueError(
            f"Incorrect normalization: {normalization}. The supported normalization strategies "
            "are `true`, `pred` and `all`"
        )

    def reset(self) -> None:
        r"""Resets the confusion matrix.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.num_predictions
            6
            >>> confmat.reset()
            >>> confmat.num_predictions
            0
        """
        self._matrix.zero_()
        self._num_predictions = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the confusion matrix with new predictions.

        Args:
        ----
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix()
            >>> confmat.update(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  2                ┃ [FP]  0                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  3                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=6
        """
        self._matrix += (
            torch.bincount(
                (target.flatten() * self.num_classes + prediction.flatten()).long(),
                minlength=self.num_classes**2,
            )
            .reshape(self.num_classes, self.num_classes)
            .to(device=self._matrix.device)
        )
        self._num_predictions = self._compute_num_predictions()

    def _compute_num_predictions(self) -> int:
        return self._matrix.sum().item()


class BinaryConfusionMatrix(BaseConfusionMatrix):
    r"""Implements a confusion matrix for binary labels.

    Args:
    ----
        matrix (``torch.Tensor`` of type long and shape ``(2, 2)``):
            Specifies the initial confusion matrix values.
            The structure of the matrix is:

                    predicted label
                        TN | FP
            true label  -------
                        FN | TP

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters import BinaryConfusionMatrix
        >>> confmat = BinaryConfusionMatrix()
        >>> confmat
        ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual negative (0) ┃ [TN]  0                ┃ [FP]  0                ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual positive (1) ┃ [FN]  0                ┃ [TP]  0                ┃
        ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
        num_predictions=0
        >>> confmat.update(
        ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
        ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
        ... )
        >>> confmat
        ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual negative (0) ┃ [TN]  2                ┃ [FP]  0                ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  3                ┃
        ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
        num_predictions=6
        >>> confmat.matrix
        tensor([[2, 0],
                [1, 3]])
        >>> confmat.num_predictions
        6
        >>> confmat.num_classes
        2
    """

    def __init__(self, matrix: Tensor | None = None) -> None:
        if matrix is None:
            matrix = torch.zeros(2, 2, dtype=torch.long)
        if matrix.shape != (2, 2):
            raise ValueError(
                f"Incorrect shape. Expected a (2, 2) matrix but received {matrix.shape}"
            )
        super().__init__(matrix)

    def __repr__(self) -> str:
        return "\n".join(
            [
                str_binary_confusion_matrix(self._matrix),
                f"num_predictions={self.num_predictions:,}",
            ]
        )

    def clone(self) -> BinaryConfusionMatrix:
        r"""Creates a copy of the current confusion matrix meter.

        Returns
        -------
            ``BinaryConfusionMatrix``: A copy of the current confusion
                matrix meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat_cloned = confmat.clone()
            >>> confmat.update(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]), target=torch.tensor([0, 1, 1, 0, 0, 1])
            ... )
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  4                ┃ [FP]  1                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  6                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=12
            >>> confmat_cloned
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  2                ┃ [FP]  0                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  3                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=6
        """
        return BinaryConfusionMatrix(self.matrix.clone())

    def equal(self, other: Any) -> bool:
        r"""Indicates if two confusion matrices are equal or not.

        Args:
        ----
            other: Specifies the value to compare.

        Returns:
        -------
            bool: ``True`` if the confusion matrices are equal,
                ``False`` otherwise.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.equal(confmat2)
            False
        """
        if not isinstance(other, BinaryConfusionMatrix):
            return False
        return self.matrix.equal(other.matrix)

    @classmethod
    def from_predictions(cls, prediction: Tensor, target: Tensor) -> BinaryConfusionMatrix:
        r"""Creates a confusion matrix given ground truth and predicted
        labels.

        Args:
        ----
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  2                ┃ [FP]  0                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  3                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=6
        """
        confmat = cls()
        confmat.update(prediction, target)
        return confmat

    ##########################
    #     Transformation     #
    ##########################

    def __add__(self, other: Any) -> BinaryConfusionMatrix:
        return self.add(other)

    def __iadd__(self, other: Any) -> BinaryConfusionMatrix:
        self.add_(other)
        return self

    def __sub__(self, other: Any) -> BinaryConfusionMatrix:
        return self.sub(other)

    def add(self, other: BinaryConfusionMatrix) -> BinaryConfusionMatrix:
        r"""Adds a confusion matrix.

        Args:
        ----
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Returns:
        -------
            ``BinaryConfusionMatrix``: A new confusion matrix
                containing the addition of the two confusion matrices.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat1 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat = confmat1.add(confmat2)
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  4                ┃ [FP]  1                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  6                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=12
        """
        check_op_compatibility_binary(self, other, "add")
        return BinaryConfusionMatrix(self.matrix.add(other.matrix))

    def add_(self, other: BinaryConfusionMatrix) -> None:
        r"""Adds a confusion matrix.

        In-place version of ``add``.

        Args:
        ----
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.add_(confmat2)
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  4                ┃ [FP]  1                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  6                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=12
        """
        check_op_compatibility_binary(self, other, "add")
        self.matrix.add_(other.matrix)
        self._num_predictions = self._compute_num_predictions()

    def merge(self, meters: Iterable[BinaryConfusionMatrix]) -> BinaryConfusionMatrix:
        r"""Merges several meters with the current meter and returns a
        new meter.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
        -------
            ``BinaryConfusionMatrix``: The merged meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat1 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat = confmat.merge([confmat1, confmat2])
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  6                ┃ [FP]  1                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  2                ┃ [TP]  9                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=18
        """
        output = self.clone()
        for meter in meters:
            output.add_(meter)
        return output

    def merge_(self, meters: Iterable[BinaryConfusionMatrix]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat1 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.merge_([confmat1, confmat2])
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  6                ┃ [FP]  1                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  2                ┃ [TP]  9                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=18
        """
        for meter in meters:
            self.add_(meter)

    def sub(self, other: BinaryConfusionMatrix) -> BinaryConfusionMatrix:
        r"""Subtracts a confusion matrix.

        Args:
        ----
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to subtract.

        Returns:
        -------
            ``BinaryConfusionMatrix``: A new confusion matrix
                containing the difference of the two confusion
                matrices.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat1 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat2 = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([1, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([0, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat = confmat1.sub(confmat2)
            >>> confmat
            ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual negative (0) ┃ [TN]  2                ┃ [FP]  0                ┃
            ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
            ┃ actual positive (1) ┃ [FN]  1                ┃ [TP]  3                ┃
            ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
            num_predictions=6
        """
        check_op_compatibility_binary(self, other, "sub")
        return BinaryConfusionMatrix(self.matrix.sub(other.matrix))

    ###################
    #     Metrics     #
    ###################

    @property
    def false_negative(self) -> int:
        r"""``int``: The false negative i.e. the number of incorrectly
        classified negative examples.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.false_negative
            1
        """
        return self._matrix[1, 0].item()

    @property
    def false_positive(self) -> int:
        r"""``int``: The false positive i.e. the number of incorrectly
        classified positive examples.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.false_positive
            0
        """
        return self._matrix[0, 1].item()

    @property
    def negative(self) -> int:
        r"""``int``: The number of negative true labels.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.negative
            2
        """
        return self.true_negative + self.false_positive

    @property
    def positive(self) -> int:
        r"""``int``: The number of positive true labels.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.positive
            4
        """
        return self.true_positive + self.false_negative

    @property
    def predictive_negative(self) -> int:
        r"""``int``: The number of negative predictions.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.predictive_negative
            3
        """
        return self.false_negative + self.true_negative

    @property
    def predictive_positive(self) -> int:
        r"""``int``: The number of positive predictions.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.predictive_positive
            3
        """
        return self.true_positive + self.false_positive

    @property
    def true_negative(self) -> int:
        r"""``int``: The true negative i.e. the number of correctly
        classified negative examples.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.true_negative
            2
        """
        return self._matrix[0, 0].item()

    @property
    def true_positive(self) -> int:
        r"""``int``: The true positive i.e. the number of correctly
        classified positive examples.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.true_positive
            3
        """
        return self._matrix[1, 1].item()

    def accuracy(self) -> float:
        r"""Computes the accuracy.

        Returns
        -------
            float: The accuracy.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.accuracy()
            0.833333...
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the accuracy because the confusion matrix is empty"
            )
        return float(self.true_positive + self.true_negative) / float(self._num_predictions)

    def balanced_accuracy(self) -> float:
        r"""Computes the balanced accuracy.

        Returns
        -------
            float: The balanced accuracy.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.balanced_accuracy()
            0.875
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the balanced accuracy because the confusion matrix "
                "is empty"
            )
        return (self.true_negative_rate() + self.true_positive_rate()) / 2

    def f_beta_score(self, beta: int | float = 1.0) -> float:
        r"""Computes the F-beta score.

        Args:
        ----
            beta (int or float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
        -------
            float: the F-beta score.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.f_beta_score()
            0.857142...
            >>> confmat.f_beta_score(2)
            0.789473...
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the F-beta score because the confusion matrix "
                "is empty"
            )
        beta2 = beta**2
        if self.true_positive == 0:
            return 0.0
        return ((1.0 + beta2) * self.true_positive) / (
            (1.0 + beta2) * self.true_positive + beta2 * self.false_negative + self.false_positive
        )

    def false_negative_rate(self) -> float:
        r"""Computes the false negative rate i.e. the miss rate.

        Returns
        -------
            float: The false negative rate.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.false_negative_rate()
            0.25
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the false negative rate because the confusion "
                "matrix is empty"
            )
        if self.positive == 0:
            return 0.0
        return float(self.false_negative) / float(self.positive)

    def false_positive_rate(self) -> float:
        r"""Computes the false positive rate i.e. the probability of
        false alarm.

        Returns
        -------
            float: The false positive rate.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.false_positive_rate()
            0.0
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the false positive rate because the confusion "
                "matrix is empty"
            )
        if self.negative == 0:
            return 0.0
        return float(self.false_positive) / float(self.negative)

    def jaccard_index(self) -> float:
        r"""Computes the Jaccard index.

        Returns
        -------
            float: The Jaccard index.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.jaccard_index()
            0.75
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the Jaccard index because the confusion "
                "matrix is empty"
            )
        if self.true_positive == 0:
            return 0.0
        return float(self.true_positive) / float(
            self.true_positive + self.false_negative + self.false_positive
        )

    def precision(self) -> float:
        r"""Computes the precision.

        Returns
        -------
            float: The precision.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.precision()
            1.0
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the precision because the confusion "
                "matrix is empty"
            )
        if self.predictive_positive == 0:
            return 0.0
        return float(self.true_positive) / float(self.predictive_positive)

    def recall(self) -> float:
        r"""Computes the recall i.e. the probability of positive
        detection.

        Returns
        -------
            float: The recall.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.recall()
            0.75
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the recall because the confusion matrix is empty"
            )
        if self.positive == 0:
            return 0.0
        return float(self.true_positive) / float(self.positive)

    def true_negative_rate(self) -> float:
        r"""Computes the true negative rate.

        Returns
        -------
            float: The true negative rate.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.true_negative_rate()
            1.0
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the true negative rate because the confusion "
                "matrix is empty"
            )
        if self.negative == 0:
            return 0.0
        return float(self.true_negative) / float(self.negative)

    def true_positive_rate(self) -> float:
        r"""Computes the true positive rate.

        Returns
        -------
            float: The true positive rate.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.true_positive_rate()
            0.75
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the true positive rate because the confusion "
                "matrix is empty"
            )
        return self.recall()

    def compute_all_metrics(
        self,
        betas: Sequence[int | float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import BinaryConfusionMatrix
            >>> confmat = BinaryConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 1, 0, 0, 1]),
            ...     target=torch.tensor([1, 1, 1, 0, 0, 1]),
            ... )
            >>> confmat.compute_all_metrics()
            {'accuracy': 0.833333...,
             'balanced_accuracy': 0.875,
             'false_negative_rate': 0.25,
             'false_negative': 1,
             'false_positive_rate': 0.0,
             'false_positive': 0,
             'jaccard_index': 0.75,
             'num_predictions': 6,
             'precision': 1.0,
             'recall': 0.75,
             'true_negative_rate': 1.0,
             'true_negative': 2,
             'true_positive_rate': 0.75,
             'true_positive': 3,
             'f1_score': 0.857142...}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the metrics because the confusion matrix is empty"
            )
        metrics = {
            f"{prefix}accuracy{suffix}": self.accuracy(),
            f"{prefix}balanced_accuracy{suffix}": self.balanced_accuracy(),
            f"{prefix}false_negative_rate{suffix}": self.false_negative_rate(),
            f"{prefix}false_negative{suffix}": self.false_negative,
            f"{prefix}false_positive_rate{suffix}": self.false_positive_rate(),
            f"{prefix}false_positive{suffix}": self.false_positive,
            f"{prefix}jaccard_index{suffix}": self.jaccard_index(),
            f"{prefix}num_predictions{suffix}": self.num_predictions,
            f"{prefix}precision{suffix}": self.precision(),
            f"{prefix}recall{suffix}": self.recall(),
            f"{prefix}true_negative_rate{suffix}": self.true_negative_rate(),
            f"{prefix}true_negative{suffix}": self.true_negative,
            f"{prefix}true_positive_rate{suffix}": self.true_positive_rate(),
            f"{prefix}true_positive{suffix}": self.true_positive,
        }
        for beta in betas:
            metrics[f"{prefix}f{beta}_score{suffix}"] = self.f_beta_score(beta)
        return metrics


class MulticlassConfusionMatrix(BaseConfusionMatrix):
    r"""Implements a confusion matrix for multiclass labels.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
        >>> confmat = MulticlassConfusionMatrix.from_predictions(
        ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
        ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
        ... )
        >>> confmat.matrix
        tensor([[2, 1, 0],
                [0, 0, 0],
                [1, 1, 1]])
        >>> confmat.num_predictions
        6
        >>> confmat.num_classes
        3
    """

    def auto_update(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the confusion matrix with new predictions.

        Unlike ``update``, this method will update the number of
        classes if a larger number of classes if found. This method
        allows to use confusion matrix in the setting where the number
        of classes is unknown at the beginning of the process.

        Args:
        ----
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.matrix
            tensor([[2, 1, 0],
                    [0, 0, 0],
                    [1, 1, 1]])
            >>> confmat.auto_update(
            ...     prediction=torch.tensor([2, 3, 2, 1, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 3, 3, 3]),
            ... )
        """
        # +1 because it is 0-indexed
        num_classes = max(prediction.max().item(), target.max().item()) + 1
        if num_classes > self.num_classes:
            self.resize(num_classes)
        self.update(prediction, target)

    def clone(self) -> MulticlassConfusionMatrix:
        r"""Creates a copy of the current confusion matrix meter.

        Returns
        -------
            ``MulticlassConfusionMatrix``: A copy of the current
                confusion matrix meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat_cloned = confmat.clone()
            >>> confmat.update(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat.matrix
            tensor([[4, 1, 1],
                    [1, 0, 1],
                    [1, 1, 2]])
            >>> confmat_cloned.matrix
            tensor([[2, 1, 0],
                    [0, 0, 0],
                    [1, 1, 1]])
        """
        return MulticlassConfusionMatrix(self.matrix.clone())

    def equal(self, other: Any) -> bool:
        r"""Indicates if two confusion matrices are equal or not.

        Args:
        ----
            other: Specifies the value to compare.

        Returns:
        -------
            bool: ``True`` if the confusion matrices are equal,
                ``False`` otherwise.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat1 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat1.equal(confmat2)
            False
        """
        if not isinstance(other, MulticlassConfusionMatrix):
            return False
        return self.matrix.equal(other.matrix)

    def resize(self, num_classes: int) -> None:
        r"""Resizes the current confusion matrix to a larger number of
        classes.

        Args:
        ----
            num_classes (int): Specifies the new number of classes.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.matrix
            tensor([[2, 1, 0],
                    [0, 0, 0],
                    [1, 1, 1]])
            >>> confmat.resize(5)
            >>> confmat.matrix
            tensor([[2, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
        """
        if num_classes < self.num_classes:
            raise ValueError(
                f"Incorrect number of classes: {num_classes}. The confusion matrix "
                f"(num_classes={self.num_classes}) can be resized only to a larger number "
                "of classes"
            )
        matrix = self._matrix
        self._matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self._matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

    @classmethod
    def from_num_classes(cls, num_classes: int) -> MulticlassConfusionMatrix:
        r"""Creates a confusion matrix given the number of classes.

        Args:
        ----
            num_classes (int): Specifies the number of classes.

        Returns:
        -------
            ``MulticlassConfusionMatrix``: An instantiated confusion
                matrix.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_num_classes(5)
            >>> confmat.matrix
            tensor([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
        """
        if num_classes < 1:
            raise ValueError(
                "Incorrect number of classes. `num_classes` has to be greater or equal to 1 but "
                f"received {num_classes}"
            )
        return cls(matrix=torch.zeros(num_classes, num_classes, dtype=torch.long))

    @classmethod
    def from_predictions(cls, prediction: Tensor, target: Tensor) -> MulticlassConfusionMatrix:
        r"""Creates a confusion matrix given ground truth and predicted
        labels.

        Note: the number of classes is inferred from the maximum
        ground truth and predicted labels.

        Args:
        ----
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.

        Returns:
        -------
            ``MulticlassConfusionMatrix``: An instantiated confusion matrix.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.matrix
            tensor([[2, 1, 0],
                    [0, 0, 0],
                    [1, 1, 1]])
        """
        # use a fake number of classes. `auto_update` will find the right number of classes
        confmat = cls.from_num_classes(num_classes=1)
        confmat.auto_update(prediction, target)
        return confmat

    ##########################
    #     Transformation     #
    ##########################

    def __add__(self, other: Any) -> MulticlassConfusionMatrix:
        return self.add(other)

    def __iadd__(self, other: Any) -> MulticlassConfusionMatrix:
        self.add_(other)
        return self

    def __sub__(self, other: Any) -> MulticlassConfusionMatrix:
        return self.sub(other)

    def add(self, other: MulticlassConfusionMatrix) -> MulticlassConfusionMatrix:
        r"""Adds a confusion matrix.

        Args:
        ----
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Returns:
        -------
            ``MulticlassConfusionMatrix``: A new confusion matrix
                containing the addition of the two confusion matrices.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat1 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat = confmat1.add(confmat2)
            >>> confmat.matrix
            tensor([[4, 1, 1],
                    [1, 0, 1],
                    [1, 1, 2]])
        """
        check_op_compatibility_multiclass(self, other, "add")
        return MulticlassConfusionMatrix(self.matrix.add(other.matrix))

    def add_(self, other: MulticlassConfusionMatrix) -> None:
        r"""Adds a confusion matrix.

        In-place version of ``add``.

        Args:
        ----
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat.add_(confmat2)
            >>> confmat.matrix
            tensor([[4, 1, 1],
                    [1, 0, 1],
                    [1, 1, 2]])
        """
        check_op_compatibility_multiclass(self, other, "add")
        self.matrix.add_(other.matrix)
        self._num_predictions = self._compute_num_predictions()

    def merge(self, meters: Iterable[MulticlassConfusionMatrix]) -> MulticlassConfusionMatrix:
        r"""Merges several meters with the current meter and returns a
        new meter.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
        -------
            ``MulticlassConfusionMatrix``: The merged meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat1 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat = confmat.merge([confmat1, confmat2])
            >>> confmat.matrix
            tensor([[6, 2, 1],
                    [1, 0, 1],
                    [2, 2, 3]])
        """
        output = self.clone()
        for meter in meters:
            output.add_(meter)
        return output

    def merge_(self, meters: Iterable[MulticlassConfusionMatrix]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
        ----
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat1 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat.merge_([confmat1, confmat2])
            >>> confmat.matrix
            tensor([[6, 2, 1],
                    [1, 0, 1],
                    [2, 2, 3]])
        """
        for meter in meters:
            self.add_(meter)

    def sub(self, other: MulticlassConfusionMatrix) -> MulticlassConfusionMatrix:
        r"""Subtracts a confusion matrix.

        Args:
        ----
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to subtract.

        Returns:
        -------
            ``MulticlassConfusionMatrix``: A new confusion matrix
                containing the difference of the two confusion matrices.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat1 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat2 = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([2, 2, 2, 0, 0, 0]),
            ...     target=torch.tensor([0, 1, 2, 0, 0, 1]),
            ... )
            >>> confmat = confmat1.sub(confmat2)
            >>> confmat.matrix
            tensor([[2, 1, 0],
                    [0, 0, 0],
                    [1, 1, 1]])
        """
        check_op_compatibility_multiclass(self, other, "sub")
        return MulticlassConfusionMatrix(self.matrix.sub(other.matrix))

    ###################
    #     Metrics     #
    ###################

    @property
    def false_negative(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The number of false negative for each class i.e. the elements
        that have been labelled as negative by the model, but they are
        actually positive.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.false_negative
            tensor([1, 0, 2])
        """
        return self.support - self.true_positive

    @property
    def false_positive(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The number of false positive for each class i.e. the elements
        that have been labelled as positive by the model, but they are
        actually negative.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.false_positive
            tensor([1, 2, 0])
        """
        return self.matrix.sum(dim=0) - self.true_positive

    @property
    def support(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The support for each class i.e. the number of elements for a
        given class (true label).

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.support
            tensor([3, 0, 3])
        """
        return self.matrix.sum(dim=1)

    @property
    def true_positive(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The number of true positive for each class i.e. the elements
        that have been labelled as positive by the model, and they are
        actually positive.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.true_positive
            tensor([2, 0, 1])
        """
        return self.matrix.diag()

    def accuracy(self) -> float:
        r"""Computes the accuracy.

        Returns
        -------
            float: The accuracy.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.accuracy()
            0.5
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the accuracy because the confusion matrix is empty"
            )
        return float(self.true_positive.sum().item()) / float(self._num_predictions)

    def balanced_accuracy(self) -> float:
        r"""Computes the balanced accuracy.

        Returns
        -------
            float: The balanced accuracy.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.balanced_accuracy()
            0.333333...
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the balanced accuracy because the confusion "
                "matrix is empty"
            )
        return self.recall().mean().item()

    def f_beta_score(self, beta: int | float = 1.0) -> Tensor:
        r"""Computes the F-beta score for each class.

        Args:
        ----
            beta (int or float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
        -------
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The F-beta score for each class.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.f_beta_score()
            tensor([0.6667, 0.0000, 0.5000])
            >>> confmat.f_beta_score(2)
            tensor([0.6667, 0.0000, 0.3846])
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the F-beta score because the confusion matrix "
                "is empty"
            )
        beta2 = beta**2
        return (self.true_positive.mul(1.0 + beta2)) / (
            self.true_positive.mul(1.0 + beta2)
            + self.false_negative.mul(beta2)
            + self.false_positive
        )

    def macro_f_beta_score(self, beta: int | float = 1.0) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) F-beta score.

        Args:
        ----
            beta (int or float, optional): Specifies the beta value.
            Default: ``1.0``

        Returns:
        -------
            float: The macro F-beta score.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.macro_f_beta_score()
            0.388888...
            >>> confmat.macro_f_beta_score(2)
            0.350427...
        """
        return self.f_beta_score(beta).mean().item()

    def micro_f_beta_score(self, beta: float = 1.0) -> float:
        r"""Computes the micro F-beta score.

        Args:
        ----
            beta (float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
        -------
            float: The micro F-beta score.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.micro_f_beta_score()
            0.5
            >>> confmat.micro_f_beta_score(2)
            0.5
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the micro F-beta score because the confusion "
                "matrix is empty"
            )
        beta2 = beta**2
        return (
            (self.true_positive.sum().mul(1.0 + beta2))
            / (
                self.true_positive.sum().mul(1.0 + beta2)
                + self.false_negative.sum().mul(beta2)
                + self.false_positive.sum()
            )
        ).item()

    def weighted_f_beta_score(self, beta: float = 1.0) -> float:
        r"""Computes the weighted mean F-beta score.

        Args:
        ----
            beta (float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
        -------
            float: The weighted mean F-beta score.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.weighted_f_beta_score()
            0.583333...
            >>> confmat.weighted_f_beta_score(2)
            0.525641...
        """
        return self.f_beta_score(beta).mul(self.support).sum().item() / float(self._num_predictions)

    def precision(self) -> Tensor:
        r"""Computes the precision for each class.

        Returns
        -------
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The precision for each class.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.precision()
            tensor([0.6667, 0.0000, 1.0000])
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the precision because the confusion matrix is empty"
            )
        return self.true_positive.float().div(self.matrix.sum(dim=0).clamp(min=1e-8))

    def macro_precision(self) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) precision.

        Returns
        -------
            float: The macro precision.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.macro_precision()
            0.555555...
        """
        return self.precision().mean().item()

    def micro_precision(self) -> float:
        r"""Computes the micro precision.

        Returns
        -------
            float: The micro precision.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.micro_precision()
            0.5...
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the micro precision because the confusion "
                "matrix is empty"
            )
        return (
            self.true_positive.sum()
            .div(self.true_positive.sum().add(self.false_positive.sum()))
            .item()
        )

    def weighted_precision(self) -> float:
        r"""Computes the weighted mean (a.k.a. unweighted mean)
        precision.

        Returns
        -------
            float: The weighted mean precision.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.weighted_precision()
            0.833333...
        """
        return self.precision().mul(self.support).sum().item() / float(self._num_predictions)

    def recall(self) -> Tensor:
        r"""Computes the recall for each class.

        Returns
        -------
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The recall for each class.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.recall()
            tensor([0.6667, 0.0000, 0.3333])
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the recall because the confusion matrix is empty"
            )
        return self.true_positive.float().div(self.support.clamp(min=1e-8))

    def macro_recall(self) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) recall.

        Returns
        -------
            float: The macro recall.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.macro_recall()
            0.333333...
        """
        return self.recall().mean().item()

    def micro_recall(self) -> float:
        r"""Computes the micro recall.

        Returns
        -------
            float: The micro recall.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.micro_recall()
            0.5
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the micro recall because the confusion matrix is empty"
            )
        return (
            self.true_positive.sum()
            .div(self.true_positive.sum().add(self.false_negative.sum()))
            .item()
        )

    def weighted_recall(self) -> float:
        r"""Computes the weighted mean (a.k.a. unweighted mean) recall.

        Returns
        -------
            float: The weighted mean recall.

        Raises
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.weighted_precision()
            0.833333...
        """
        return self.recall().mul(self.support).sum().item() / float(self._num_predictions)

    def compute_per_class_metrics(
        self,
        betas: Sequence[float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, Tensor]:
        r"""Computes all the per-class metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the per-class metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.compute_per_class_metrics()
            {'precision': tensor([0.6667, 0.0000, 1.0000]),
             'recall': tensor([0.6667, 0.0000, 0.3333]),
             'f1_score': tensor([0.6667, 0.0000, 0.5000])}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the metrics because the confusion matrix is empty"
            )

        metrics = {
            f"{prefix}precision{suffix}": self.precision(),
            f"{prefix}recall{suffix}": self.recall(),
        }
        for beta in betas:
            metrics[f"{prefix}f{beta}_score{suffix}"] = self.f_beta_score(beta)
        return metrics

    def compute_macro_metrics(
        self,
        betas: Sequence[float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the "macro" metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the "macro" metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.compute_macro_metrics()
            {'macro_precision': 0.555555...,
             'macro_recall': 0.333333...,
             'macro_f1_score': 0.388888...}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the 'macro' metrics because the confusion "
                "matrix is empty"
            )
        metrics = {
            f"{prefix}macro_precision{suffix}": self.macro_precision(),
            f"{prefix}macro_recall{suffix}": self.macro_recall(),
        }
        for beta in betas:
            metrics[f"{prefix}macro_f{beta}_score{suffix}"] = self.macro_f_beta_score(beta)
        return metrics

    def compute_micro_metrics(
        self,
        betas: Sequence[float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the "micro" metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the "micro" metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.compute_micro_metrics()
            {'micro_precision': 0.5,
             'micro_recall': 0.5,
             'micro_f1_score': 0.5}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the 'micro' metrics because the confusion "
                "matrix is empty"
            )
        metrics = {
            f"{prefix}micro_precision{suffix}": self.micro_precision(),
            f"{prefix}micro_recall{suffix}": self.micro_recall(),
        }
        for beta in betas:
            metrics[f"{prefix}micro_f{beta}_score{suffix}"] = self.micro_f_beta_score(beta)
        return metrics

    def compute_weighted_metrics(
        self,
        betas: Sequence[float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the "weighted" metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the "weighted" metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.compute_weighted_metrics()
            {'weighted_precision': 0.833333...,
             'weighted_recall': 0.5,
             'weighted_f1_score': 0.583333...}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the 'weighted' metrics because the confusion "
                "matrix is empty"
            )
        metrics = {
            f"{prefix}weighted_precision{suffix}": self.weighted_precision(),
            f"{prefix}weighted_recall{suffix}": self.weighted_recall(),
        }
        for beta in betas:
            metrics[f"{prefix}weighted_f{beta}_score{suffix}"] = self.weighted_f_beta_score(beta)
        return metrics

    def compute_scalar_metrics(
        self,
        betas: Sequence[float] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the scalar metrics.

        Args:
        ----
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
        -------
            dict: All the scalar metrics.

        Raises:
        ------
            ``EmptyMeterError`` if the confusion matrix is empty.

        Example usage:

        .. code-block:: pycon

            >>> from gravitorch.utils.meters import MulticlassConfusionMatrix
            >>> confmat = MulticlassConfusionMatrix.from_predictions(
            ...     prediction=torch.tensor([0, 1, 2, 0, 0, 1]),
            ...     target=torch.tensor([2, 2, 2, 0, 0, 0]),
            ... )
            >>> confmat.compute_scalar_metrics()
            {'accuracy': 0.5,
             'balanced_accuracy': 0.333333...,
             'macro_precision': 0.555555...,
             'macro_recall': 0.333333...,
             'macro_f1_score': 0.388888...,
             'micro_precision': 0.5,
             'micro_recall': 0.5,
             'micro_f1_score': 0.5,
             'weighted_precision': 0.833333...,
             'weighted_recall': 0.5,
             'weighted_f1_score': 0.583333...}
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the metrics because the confusion matrix is empty"
            )
        metrics = {
            f"{prefix}accuracy{suffix}": self.accuracy(),
            f"{prefix}balanced_accuracy{suffix}": self.balanced_accuracy(),
        }
        metrics.update(self.compute_macro_metrics(betas, prefix, suffix))
        metrics.update(self.compute_micro_metrics(betas, prefix, suffix))
        metrics.update(self.compute_weighted_metrics(betas, prefix, suffix))
        return metrics


def check_confusion_matrix(matrix: Tensor) -> None:
    r"""Checks if the input matrix is a valid confusion matrix.

    Args:
    ----
        matrix (``torch.Tensor``): Specifies the matrix to check.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.meters.confmat import check_confusion_matrix
        >>> check_confusion_matrix(torch.zeros(3, 3, dtype=torch.long))
    """
    if matrix.ndim != 2:
        raise ValueError(
            "Incorrect matrix dimensions. The matrix must have 2 dimensions but "
            f"received {matrix.ndim} dimensions"
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Incorrect matrix shape. The matrix must be a squared matrix but "
            f"received {matrix.shape}"
        )
    if matrix.dtype != torch.long:
        raise ValueError(
            "Incorrect matrix data type. The matrix data type must be long but "
            f"received {matrix.dtype}"
        )
    if not torch.all(matrix >= 0):
        raise ValueError(
            "Incorrect matrix values. The matrix values must be greater or equal to 0 but "
            f"received:\n{matrix}"
        )


def check_op_compatibility_binary(
    current: BinaryConfusionMatrix, other: BinaryConfusionMatrix, op_name: str
) -> None:
    r"""Checks if the confusion matrices for binary labels are
    compatible.

    Args:
    ----
        current (``BinaryConfusionMatrix``): Specifies the current
            confusion matrix for binary labels.
        other (``BinaryConfusionMatrix``): Specifies the other
            confusion matrix for binary labels.
        op_name (str): Specifies the operation name.

    Raises:
    ------
        TypeError if the other matrix type is not compatible.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters.confmat import (
        ...     BinaryConfusionMatrix,
        ...     check_op_compatibility_binary,
        ... )
        >>> check_op_compatibility_binary(
        ...     BinaryConfusionMatrix(), BinaryConfusionMatrix(), op_name="add"
        ... )
    """
    if not isinstance(other, BinaryConfusionMatrix):
        raise TypeError(
            f"Incorrect type {type(other)}. No implementation available to `{op_name}` "
            f"{type(current)} with {type(other)}"
        )


def check_op_compatibility_multiclass(
    current: MulticlassConfusionMatrix, other: MulticlassConfusionMatrix, op_name: str
) -> None:
    r"""Checks if the confusion matrices for multiclass labels are
    compatible.

    Args:
    ----
        current (``MulticlassConfusionMatrix``): Specifies the current
            confusion matrix for multiclass labels.
        other (``MulticlassConfusionMatrix``): Specifies the other
            confusion matrix for multiclass labels.
        op_name (str): Specifies the operation name.

    Raises:
    ------
        TypeError if the other matrix type is not compatible.
        ValueError if the matrix shapes are different.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.utils.meters.confmat import (
        ...     MulticlassConfusionMatrix,
        ...     check_op_compatibility_multiclass,
        ... )
        >>> check_op_compatibility_multiclass(
        ...     MulticlassConfusionMatrix.from_num_classes(5),
        ...     MulticlassConfusionMatrix.from_num_classes(5),
        ...     op_name="add",
        ... )
    """
    if not isinstance(other, MulticlassConfusionMatrix):
        raise TypeError(
            f"Incorrect type: {type(other)}. No implementation available to `{op_name}` "
            f"{type(current)} with {type(other)}"
        )
    if current.matrix.shape != other.matrix.shape:
        raise ValueError(
            f"Incorrect shape: received {other.matrix.shape} but expect {current.matrix.shape}"
        )


def str_binary_confusion_matrix(confmat: Tensor) -> str:
    r"""Computes a string representation of the confusion matrix.

    Args:
        confmat (``torch.Tensor``): Specifies the confusion matrix.

    Returns:
        str: A string representation of the confusion matrix.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.utils.meters.confmat import str_binary_confusion_matrix
        >>> print(str_binary_confusion_matrix(torch.tensor([[1001, 42], [123, 789]])))
        ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                     ┃ predicted negative (0) ┃ predicted positive (1) ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual negative (0) ┃ [TN]  1,001            ┃ [FP]  42               ┃
        ┣━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━┫
        ┃ actual positive (1) ┃ [FN]  123              ┃ [TP]  789              ┃
        ┗━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━┛
    """
    if confmat.shape != (2, 2):
        raise RuntimeError(f"Expected a 2x2 confusion matrix but received: {confmat.shape}")
    confmat = confmat.long()
    table = [
        ["", "predicted negative (0)", "predicted positive (1)"],
        ["actual negative (0)", f"[TN]  {confmat[0,0]:,}", f"[FP]  {confmat[0,1]:,}"],
        ["actual positive (1)", f"[FN]  {confmat[1,0]:,}", f"[TP]  {confmat[1,1]:,}"],
    ]
    return tabulate(table, tablefmt="heavy_grid")
