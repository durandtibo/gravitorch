__all__ = ["BaseConfusionMatrix", "BinaryConfusionMatrix", "MulticlassConfusionMatrix"]

from collections.abc import Iterable, Sequence
from typing import Optional, Union

import torch
from torch import Tensor

from gravitorch.distributed.ddp import SUM, sync_reduce_
from gravitorch.utils.format import str_indent
from gravitorch.utils.meters.exceptions import EmptyMeterError


class BaseConfusionMatrix:
    r"""Defines the base class to implement confusion matrix.

    Args:
        matrix (``torch.Tensor`` of type long and shape
            ``(num_classes, num_classes)``): Specifies the initial
            confusion matrix values. The rows indicate the true
            labels and the columns indicate the predicted labels.
    """

    def __init__(self, matrix: Tensor):
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
        num_classes)``: The confusion matrix values."""
        return self._matrix

    @property
    def num_classes(self) -> int:
        r"""int: The number of classes."""
        return self._matrix.shape[0]

    @property
    def num_predictions(self) -> int:
        r"""int: The number of predictions."""
        return self._num_predictions

    def all_reduce(self) -> None:
        r"""Reduces the meter values across all machines in such a way that all
        get the final result.

        The confusion matrix is reduced by summing all the confusion
        matrices (1 confusion matrix per distributed process).
        """
        sync_reduce_(self._matrix, SUM)
        # It is necessary to recompute the number of predictions because
        # the confusion matrix may have changed
        self._num_predictions = self._compute_num_predictions()

    def get_normalized_matrix(self, normalization: str) -> Tensor:
        r"""Gets the normalized confusion matrix.

        Args:
            normalization (str): Specifies the normalization strategy.
                The supported normalization strategies are:

                    - ``'true'``: normalization over the targets
                        (most commonly used)
                    - ``'pred'``: normalization over the predictions
                    - ``'all'``: normalization over the whole matrix

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(num_classes, num_classes)``: The normalized
                confusion matrix.

        Raises:
            ValueError if the normalization strategy is not supported.
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
            f"Incorrect normalization {normalization}. The supported normalization strategies "
            "are `true`, `pred` and `all`"
        )

    def reset(self) -> None:
        r"""Resets the confusion matrix."""
        self._matrix.zero_()
        self._num_predictions = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the confusion matrix with new predictions.

        Args:
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.
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
        matrix (``torch.Tensor`` of type long and shape ``(2, 2)``):
            Specifies the initial confusion matrix values.
    """

    def __init__(self, matrix: Optional[Tensor] = None):
        if matrix is None:
            matrix = torch.zeros(2, 2, dtype=torch.long)
        if matrix.shape != (2, 2):
            raise ValueError
        super().__init__(matrix)

    def clone(self) -> "BinaryConfusionMatrix":
        r"""Creates a copy of the current confusion matrix meter.

        Returns:
            ``BinaryConfusionMatrix``: A copy of the current confusion
                matrix meter.
        """
        return BinaryConfusionMatrix(self.matrix.clone())

    def equal(self, other) -> bool:
        r"""Indicates if two confusion matrices are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the confusion matrices are equal,
                ``False`` otherwise.
        """
        if not isinstance(other, BinaryConfusionMatrix):
            return False
        return self.matrix.equal(other.matrix)

    @classmethod
    def from_predictions(cls, prediction: Tensor, target: Tensor) -> "BinaryConfusionMatrix":
        r"""Creates a confusion matrix given ground truth and predicted labels.

        Args:
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.
        """
        confmat = cls()
        confmat.update(prediction, target)
        return confmat

    ##########################
    #     Transformation     #
    ##########################

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        self.add_(other)
        return self

    def __sub__(self, other):
        return self.sub(other)

    def add(self, other: "BinaryConfusionMatrix") -> "BinaryConfusionMatrix":
        r"""Adds a confusion matrix.

        Args:
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Returns:
            ``BinaryConfusionMatrix``: A new confusion matrix
                containing the addition of the two confusion matrices.
        """
        check_op_compatibility_binary(self, other, "add")
        return BinaryConfusionMatrix(self.matrix.add(other.matrix))

    def add_(self, other: "BinaryConfusionMatrix") -> None:
        r"""Adds a confusion matrix.

        In-place version of ``add``.

        Args:
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to add.
        """
        check_op_compatibility_binary(self, other, "add")
        self.matrix.add_(other.matrix)
        self._num_predictions = self._compute_num_predictions()

    def merge(self, meters: Iterable["BinaryConfusionMatrix"]) -> "BinaryConfusionMatrix":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``BinaryConfusionMatrix``: The merged meter.
        """
        output = self.clone()
        for meter in meters:
            output.add_(meter)
        return output

    def merge_(self, meters: Iterable["BinaryConfusionMatrix"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self.add_(meter)

    def sub(self, other: "BinaryConfusionMatrix") -> "BinaryConfusionMatrix":
        r"""Subtracts a confusion matrix.

        Args:
            other (``BinaryConfusionMatrix``): Specifies the other
                confusion matrix to subtract.

        Returns:
            ``BinaryConfusionMatrix``: A new confusion matrix
                containing the difference of the two confusion
                matrices.
        """
        check_op_compatibility_binary(self, other, "sub")
        return BinaryConfusionMatrix(self.matrix.sub(other.matrix))

    ###################
    #     Metrics     #
    ###################

    @property
    def false_negative(self) -> int:
        r"""int: The false negative i.e. the number of incorrectly
        classified negative examples."""
        return self._matrix[0, 1].item()

    @property
    def false_positive(self) -> int:
        r"""int: The false positive i.e. the number of incorrectly
        classified positive examples."""
        return self._matrix[1, 0].item()

    @property
    def negative(self) -> int:
        r"""int: The number of negative true labels."""
        return self.true_negative + self.false_positive

    @property
    def positive(self) -> int:
        r"""int: The number of positive true labels."""
        return self.true_positive + self.false_negative

    @property
    def predictive_negative(self) -> int:
        r"""int: The number of negative predictions."""
        return self.false_negative + self.true_negative

    @property
    def predictive_positive(self) -> int:
        r"""int: The number of positive predictions."""
        return self.true_positive + self.false_positive

    @property
    def true_negative(self) -> int:
        r"""int: The true negative i.e. the number of correctly
        classified negative examples."""
        return self._matrix[1, 1].item()

    @property
    def true_positive(self) -> int:
        r"""int: The true positive i.e. the number of correctly
        classified positive examples."""
        return self._matrix[0, 0].item()

    def accuracy(self) -> float:
        r"""Computes the accuracy.

        Returns:
            float: The accuracy.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the accuracy because the confusion matrix is empty"
            )
        return float(self.true_positive + self.true_negative) / float(self._num_predictions)

    def balanced_accuracy(self) -> float:
        r"""Computes the balanced accuracy.

        Returns:
            float: The balanced accuracy.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the balanced accuracy because the confusion matrix "
                "is empty"
            )
        return (self.true_negative_rate() + self.true_positive_rate()) / 2

    def f_beta_score(self, beta: float = 1.0) -> float:
        r"""Computes the F-beta score.

        Args:
            beta (float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
            float: the F-beta score.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the F-beta score because the confusion matrix "
                "is empty"
            )
        beta2 = beta**2
        return ((1.0 + beta2) * self.true_positive) / (
            (1.0 + beta2) * self.true_positive + beta2 * self.false_negative + self.false_positive
        )

    def false_negative_rate(self) -> float:
        r"""Computes the false negative rate i.e. the miss rate.

        Returns:
            float: The false negative rate.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the false negative rate because the confusion "
                "matrix is empty"
            )
        return float(self.false_negative) / float(self.positive)

    def false_positive_rate(self) -> float:
        r"""Computes the false positive rate i.e. the probability of false
        alarm.

        Returns:
            float: The false positive rate.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the false positive rate because the confusion "
                "matrix is empty"
            )
        return float(self.false_positive) / float(self.negative)

    def jaccard_index(self) -> float:
        r"""Computes the Jaccard index.

        Returns:
            float: The Jaccard index.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the Jaccard index because the confusion "
                "matrix is empty"
            )
        return float(self.true_positive) / float(
            self.true_positive + self.false_negative + self.false_positive
        )

    def precision(self) -> float:
        r"""Computes the precision.

        Returns:
            float: The precision.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the precision because the confusion "
                "matrix is empty"
            )
        return float(self.true_positive) / float(self.predictive_positive)

    def recall(self) -> float:
        r"""Computes the recall i.e. the probability of positive detection.

        Returns:
            float: The recall.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the recall because the confusion " "matrix is empty"
            )
        return float(self.true_positive) / float(self.positive)

    def true_negative_rate(self) -> float:
        r"""Computes the true negative rate.

        Returns:
            float: The true negative rate.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the true negative rate because the confusion "
                "matrix is empty"
            )
        return float(self.true_negative) / float(self.negative)

    def true_positive_rate(self) -> float:
        r"""Computes the true positive rate.

        Returns:
            float: The true positive rate.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the true positive rate because the confusion "
                "matrix is empty"
            )
        return self.recall()

    def compute_all_metrics(
        self,
        betas: Sequence[Union[int, float]] = (1,),
        prefix: str = "",
        suffix: str = "",
    ) -> dict[str, float]:
        r"""Computes all the metrics.

        Args:
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the metrics because the confusion matrix is empty"
            )
        metrics = {
            f"{prefix}accuracy{suffix}": self.accuracy(),
            f"{prefix}balanced_accuracy{suffix}": self.balanced_accuracy(),
            f"{prefix}false_negative_rate{suffix}": self.false_negative_rate(),
            f"{prefix}false_positive_rate{suffix}": self.false_positive_rate(),
            f"{prefix}jaccard_index{suffix}": self.jaccard_index(),
            f"{prefix}precision{suffix}": self.precision(),
            f"{prefix}recall{suffix}": self.recall(),
            f"{prefix}true_negative_rate{suffix}": self.true_negative_rate(),
            f"{prefix}true_positive_rate{suffix}": self.true_positive_rate(),
        }
        for beta in betas:
            metrics[f"{prefix}f{beta}_score{suffix}"] = self.f_beta_score(beta)
        return metrics


class MulticlassConfusionMatrix(BaseConfusionMatrix):
    r"""Implements a confusion matrix for multiclass labels."""

    def auto_update(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the confusion matrix with new predictions.

        Unlike ``update``, this method will update the number of
        classes if a larger number of classes if found. This method
        allows to use confusion matrix in the setting where the number
        of classes is unknown at the beginning of the process.

        Args:
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.
        """
        # +1 because it is 0-indexed
        num_classes = max(prediction.max().item(), target.max().item()) + 1
        if num_classes > self.num_classes:
            self.resize(num_classes)
        self.update(prediction, target)

    def clone(self) -> "MulticlassConfusionMatrix":
        r"""Creates a copy of the current confusion matrix meter.

        Returns:
            ``MulticlassConfusionMatrix``: A copy of the current confusion matrix meter.
        """
        return MulticlassConfusionMatrix(self.matrix.clone())

    def equal(self, other) -> bool:
        r"""Indicates if two confusion matrices are equal or not.

        Args:
            other: Specifies the value to compare.

        Returns:
            bool: ``True`` if the confusion matrices are equal, ``False`` otherwise.
        """
        if not isinstance(other, MulticlassConfusionMatrix):
            return False
        return self.matrix.equal(other.matrix)

    def resize(self, num_classes: int) -> None:
        r"""Resizes the current confusion matrix to a larger number of classes.

        Args:
            num_classes (int): Specifies the new number of classes.
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
    def from_num_classes(cls, num_classes: int) -> "MulticlassConfusionMatrix":
        r"""Creates a confusion matrix given the number of classes.

        Args:
            num_classes (int): Specifies the number of classes.

        Returns:
            ``MulticlassConfusionMatrix``: An instantiated confusion
                matrix.
        """
        if num_classes < 1:
            raise ValueError(
                "Incorrect `num_classes`. `num_classes` has to be greater or equal to 1 but "
                f"received {num_classes}"
            )
        return cls(matrix=torch.zeros(num_classes, num_classes, dtype=torch.long))

    @classmethod
    def from_predictions(cls, prediction: Tensor, target: Tensor) -> "MulticlassConfusionMatrix":
        r"""Creates a confusion matrix given ground truth and predicted labels.

        Note: the number of classes is inferred from the maximum
        ground truth and predicted labels.

        Args:
            prediction (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the predicted labels.
            target (``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``): Specifies the ground truth
                labels.

        Returns:
            ``MulticlassConfusionMatrix``: An instantiated confusion matrix.
        """
        # use a fake number of classes. `auto_update` will find the right number of classes
        confmat = cls.from_num_classes(num_classes=1)
        confmat.auto_update(prediction, target)
        return confmat

    ##########################
    #     Transformation     #
    ##########################

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        self.add_(other)
        return self

    def __sub__(self, other):
        return self.sub(other)

    def add(self, other: "MulticlassConfusionMatrix") -> "MulticlassConfusionMatrix":
        r"""Adds a confusion matrix.

        Args:
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to add.

        Returns:
            ``MulticlassConfusionMatrix``: A new confusion matrix
                containing the addition of the two confusion matrices.
        """
        check_op_compatibility_multiclass(self, other, "add")
        return MulticlassConfusionMatrix(self.matrix.add(other.matrix))

    def add_(self, other: "MulticlassConfusionMatrix") -> None:
        r"""Adds a confusion matrix.

        In-place version of ``add``.

        Args:
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to add.
        """
        check_op_compatibility_multiclass(self, other, "add")
        self.matrix.add_(other.matrix)
        self._num_predictions = self._compute_num_predictions()

    def merge(self, meters: Iterable["MulticlassConfusionMatrix"]) -> "MulticlassConfusionMatrix":
        r"""Merges several meters with the current meter and returns a new
        meter.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.

        Returns:
            ``MulticlassConfusionMatrix``: The merged meter.
        """
        output = self.clone()
        for meter in meters:
            output.add_(meter)
        return output

    def merge_(self, meters: Iterable["MulticlassConfusionMatrix"]) -> None:
        r"""Merges several meters into the current meter.

        In-place version of ``merge``.

        Args:
            meters (iterable): Specifies the meters to merge to the
                current meter.
        """
        for meter in meters:
            self.add_(meter)

    def sub(self, other: "MulticlassConfusionMatrix") -> "MulticlassConfusionMatrix":
        r"""Subtracts a confusion matrix.

        Args:
            other (``MulticlassConfusionMatrix``): Specifies the other
                confusion matrix to subtract.

        Returns:
            ``MulticlassConfusionMatrix``: A new confusion matrix
                containing the difference of the two confusion matrices.
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
        """
        return self.support - self.true_positive

    @property
    def false_positive(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The number of false positive for each class i.e. the elements
        that have been labelled as positive by the model, but they are
        actually negative.
        """
        return self.matrix.sum(dim=0) - self.true_positive

    @property
    def support(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The support for each class i.e. the number of elements for a
        given class (true label).
        """
        return self.matrix.sum(dim=1)

    @property
    def true_positive(self) -> Tensor:
        r"""``torch.Tensor`` of type long and shape ``(num_classes,)``:

        The number of true positive for each class i.e. the elements
        that have been labelled as positive by the model, and they are
        actually positive.
        """
        return self.matrix.diag()

    def accuracy(self) -> float:
        r"""Computes the accuracy.

        Returns:
            float: The accuracy.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the accuracy because the confusion matrix is empty"
            )
        return float(self.true_positive.sum().item()) / float(self._num_predictions)

    def balanced_accuracy(self) -> float:
        r"""Computes the balanced accuracy.

        Returns:
            float: The balanced accuracy.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the balanced accuracy because the confusion "
                "matrix is empty"
            )
        return self.recall().mean().item()

    def f_beta_score(self, beta: float = 1.0) -> Tensor:
        r"""Computes the F-beta score for each class.

        Args:
            beta (float, optional): Specifies the beta value. Default: ``1.0``

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The F-beta score for each class.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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

    def macro_f_beta_score(self, beta: float = 1.0) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) F-beta score.

        Args:
            beta (float, optional): Specifies the beta value. Default: ``1.0``

        Returns:
            float: The macro F-beta score.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        return self.f_beta_score(beta).mean().item()

    def micro_f_beta_score(self, beta: float = 1.0) -> float:
        r"""Computes the micro F-beta score.

        Args:
            beta (float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
            float: The micro F-beta score.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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
            beta (float, optional): Specifies the beta value.
                Default: ``1.0``

        Returns:
            float: The weighted mean F-beta score.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        return self.f_beta_score(beta).mul(self.support).sum().item() / float(self._num_predictions)

    def precision(self) -> Tensor:
        r"""Computes the precision for each class.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The precision for each class.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the precision because the confusion matrix is empty"
            )
        return self.true_positive.float().div(self.matrix.sum(dim=0).clamp(min=1e-8))

    def macro_precision(self) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) precision.

        Returns:
            float: The macro precision.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        return self.precision().mean().item()

    def micro_precision(self) -> float:
        r"""Computes the micro precision.

        Returns:
            float: The micro precision.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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
        r"""Computes the weighted mean (a.k.a. unweighted mean) precision.

        Returns:
            float: The weighted mean precision.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        return self.precision().mul(self.support).sum().item() / float(self._num_predictions)

    def recall(self) -> Tensor:
        r"""Computes the recall for each class.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(num_classes,)``: The recall for each class.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the recall because the confusion matrix is empty"
            )
        return self.true_positive.float().div(self.support.clamp(min=1e-8))

    def macro_recall(self) -> float:
        r"""Computes the macro (a.k.a. unweighted mean) recall.

        Returns:
            float: The macro recall.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        return self.recall().mean().item()

    def micro_recall(self) -> float:
        r"""Computes the micro recall.

        Returns:
            float: The micro recall.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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

        Returns:
            float: The weighted mean recall.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the per-class metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the "macro" metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the `macro` metrics because the confusion "
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
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the "micro" metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the `micro` metrics because the confusion "
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
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the "weighted" metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
        """
        if self.num_predictions == 0:
            raise EmptyMeterError(
                "It is not possible to compute the `weighted` metrics because the confusion "
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
            betas (sequence, optional): Specifies the betas used to
                compute the f-beta score. Default: ``(1,)``
            prefix (str, optional): Specifies a prefix for all the
                metrics. Default: ``''``
            suffix (str, optional): Specifies a suffix for all the
                metrics. Default: ``''``

        Returns:
            dict: All the scalar metrics.

        Raises:
            ``EmptyMeterError`` if the confusion matrix is empty.
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
        matrix (``torch.Tensor``): Specifies the matrix to check.
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
    r"""Checks if the confusion matrices for binary labels are compatible.

    Args:
        current (``BinaryConfusionMatrix``): Specifies the current confusion matrix for binary labels.
        other (``BinaryConfusionMatrix``): Specifies the other confusion matrix for binary labels.
        op_name (str): Specifies the operation name.

    Raises:
        TypeError if the other matrix type is not compatible.
    """
    if not isinstance(other, BinaryConfusionMatrix):
        raise TypeError(
            f"Incorrect type {type(other)}. No implementation available to `{op_name}` "
            f"{type(current)} with {type(other)}"
        )


def check_op_compatibility_multiclass(
    current: MulticlassConfusionMatrix, other: MulticlassConfusionMatrix, op_name: str
) -> None:
    r"""Checks if the confusion matrices for multiclass labels are compatible.

    Args:
        current (``MulticlassConfusionMatrix``): Specifies the current
            confusion matrix for multiclass labels.
        other (``MulticlassConfusionMatrix``): Specifies the other
            confusion matrix for multiclass labels.
        op_name (str): Specifies the operation name.

    Raises:
        TypeError if the other matrix type is not compatible.
        ValueError if the matrix shapes are different.
    """
    if not isinstance(other, MulticlassConfusionMatrix):
        raise TypeError(
            f"Incorrect type {type(other)}. No implementation available to `{op_name}` "
            f"{type(current)} with {type(other)}"
        )
    if current.matrix.shape != other.matrix.shape:
        raise ValueError(
            f"Incorrect shape: received {other.matrix.shape} but expect {current.matrix.shape}"
        )
