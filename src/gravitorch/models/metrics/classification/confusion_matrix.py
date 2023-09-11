r"""This module defines the confusion matrix metric for binary
labels."""

from __future__ import annotations

__all__ = ["BinaryConfusionMatrix", "CategoricalConfusionMatrix"]

import logging
from collections.abc import Sequence

from coola.utils import str_mapping
from torch import Tensor

from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base import EmptyMetricError
from gravitorch.models.metrics.base_epoch import BaseEpochMetric
from gravitorch.nn import ToCategoricalLabel
from gravitorch.utils.artifacts import PyTorchArtifact
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.format import str_scalar
from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory
from gravitorch.utils.meters import BinaryConfusionMatrix as BinaryConfusionMatrixMeter
from gravitorch.utils.meters import (
    MulticlassConfusionMatrix as CategoricalConfusionMatrixMeter,
)
from gravitorch.utils.tensor import str_full_tensor

logger = logging.getLogger(__name__)


class BinaryConfusionMatrix(BaseEpochMetric):
    r"""Implements the confusion matrix metric for binary labels.

    Args:
    ----
        mode (str): Specifies the mode (e.g. train or eval).
        name (str, optional): Specifies the name of the metric.
            The name is used to log the metric results.
                Default: ``'bin_conf_mat'``
        betas (sequence, optional): Specifies the betas used to
            compute the f-beta score. Default: ``(1,)``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import BinaryConfusionMatrix
        >>> metric = BinaryConfusionMatrix("eval")
        >>> metric
        BinaryConfusionMatrix(
          (mode): eval
          (name): bin_conf_mat
          (betas): (1,)
          (confusion_matrix): BinaryConfusionMatrix(num_classes=2, num_predictions=0)
        )
        >>> metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
        >>> metric.value()
        {'eval/bin_conf_mat_accuracy': 1.0,
         'eval/bin_conf_mat_balanced_accuracy': 1.0,
         'eval/bin_conf_mat_false_negative_rate': 0.0,
         'eval/bin_conf_mat_false_negative': 0,
         'eval/bin_conf_mat_false_positive_rate': 0.0,
         'eval/bin_conf_mat_false_positive': 0,
         'eval/bin_conf_mat_jaccard_index': 1.0,
         'eval/bin_conf_mat_num_predictions': 4,
         'eval/bin_conf_mat_precision': 1.0,
         'eval/bin_conf_mat_recall': 1.0,
         'eval/bin_conf_mat_true_negative_rate': 1.0,
         'eval/bin_conf_mat_true_negative': 2,
         'eval/bin_conf_mat_true_positive_rate': 1.0,
         'eval/bin_conf_mat_true_positive': 2,
         'eval/bin_conf_mat_f1_score': 1.0}
        >>> metric(torch.tensor([1, 0]), torch.tensor([1, 0]))
        >>> metric.value()
        {'eval/bin_conf_mat_accuracy': 1.0,
         'eval/bin_conf_mat_balanced_accuracy': 1.0,
         'eval/bin_conf_mat_false_negative_rate': 0.0,
         'eval/bin_conf_mat_false_negative': 0,
         'eval/bin_conf_mat_false_positive_rate': 0.0,
         'eval/bin_conf_mat_false_positive': 0,
         'eval/bin_conf_mat_jaccard_index': 1.0,
         'eval/bin_conf_mat_num_predictions': 6,
         'eval/bin_conf_mat_precision': 1.0,
         'eval/bin_conf_mat_recall': 1.0,
         'eval/bin_conf_mat_true_negative_rate': 1.0,
         'eval/bin_conf_mat_true_negative': 3,
         'eval/bin_conf_mat_true_positive_rate': 1.0,
         'eval/bin_conf_mat_true_positive': 3,
         'eval/bin_conf_mat_f1_score': 1.0}
        >>> metric.reset()
        >>> metric(torch.tensor([1, 0]), torch.tensor([1, 0]))
        >>> metric.value()
        {'eval/bin_conf_mat_accuracy': 1.0,
         'eval/bin_conf_mat_balanced_accuracy': 1.0,
         'eval/bin_conf_mat_false_negative_rate': 0.0,
         'eval/bin_conf_mat_false_negative': 0,
         'eval/bin_conf_mat_false_positive_rate': 0.0,
         'eval/bin_conf_mat_false_positive': 0,
         'eval/bin_conf_mat_jaccard_index': 1.0,
         'eval/bin_conf_mat_num_predictions': 2,
         'eval/bin_conf_mat_precision': 1.0,
         'eval/bin_conf_mat_recall': 1.0,
         'eval/bin_conf_mat_true_negative_rate': 1.0,
         'eval/bin_conf_mat_true_negative': 1,
         'eval/bin_conf_mat_true_positive_rate': 1.0,
         'eval/bin_conf_mat_true_positive': 1,
         'eval/bin_conf_mat_f1_score': 1.0}
    """

    def __init__(
        self,
        mode: str,
        name: str = "bin_conf_mat",
        betas: Sequence[int | float] = (1,),
    ) -> None:
        super().__init__(mode=mode, name=name)
        self._confusion_matrix = BinaryConfusionMatrixMeter()
        self._betas = tuple(betas)
        self.reset()

    def extra_repr(self) -> str:
        return str_mapping(
            {
                "mode": self._mode,
                "name": self._name,
                "betas": self._betas,
                "confusion_matrix": self._confusion_matrix,
            }
        )

    def attach(self, engine: BaseEngine) -> None:
        super().attach(engine)
        trackers = [
            MaxScalarHistory(name=f"{self._metric_name}_accuracy"),
            MaxScalarHistory(name=f"{self._metric_name}_balanced_accuracy"),
            MinScalarHistory(name=f"{self._metric_name}_false_negative_rate"),
            MinScalarHistory(name=f"{self._metric_name}_false_positive_rate"),
            MaxScalarHistory(name=f"{self._metric_name}_jaccard_index"),
            MaxScalarHistory(name=f"{self._metric_name}_precision"),
            MaxScalarHistory(name=f"{self._metric_name}_recall"),
            MaxScalarHistory(name=f"{self._metric_name}_true_negative_rate"),
            MaxScalarHistory(name=f"{self._metric_name}_true_positive_rate"),
        ]
        for beta in self._betas:
            trackers.append(MaxScalarHistory(name=f"{self._metric_name}_f{beta}_score"))
        for tracker in trackers:
            engine.add_history(tracker)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean absolute error metric given a mini-batch of
        examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type float or long): Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type long or float): Specifies the target tensor.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import BinaryConfusionMatrix
            >>> metric = BinaryConfusionMatrix("eval")
            >>> metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
            >>> metric.value()
            {'eval/bin_conf_mat_accuracy': 1.0,
             'eval/bin_conf_mat_balanced_accuracy': 1.0,
             'eval/bin_conf_mat_false_negative_rate': 0.0,
             'eval/bin_conf_mat_false_negative': 0,
             'eval/bin_conf_mat_false_positive_rate': 0.0,
             'eval/bin_conf_mat_false_positive': 0,
             'eval/bin_conf_mat_jaccard_index': 1.0,
             'eval/bin_conf_mat_num_predictions': 4,
             'eval/bin_conf_mat_precision': 1.0,
             'eval/bin_conf_mat_recall': 1.0,
             'eval/bin_conf_mat_true_negative_rate': 1.0,
             'eval/bin_conf_mat_true_negative': 2,
             'eval/bin_conf_mat_true_positive_rate': 1.0,
             'eval/bin_conf_mat_true_positive': 2,
             'eval/bin_conf_mat_f1_score': 1.0}
        """
        self._confusion_matrix.update(prediction.flatten(), target.flatten())

    def reset(self) -> None:
        self._confusion_matrix.reset()

    def value(self, engine: BaseEngine | None = None) -> dict:
        self._confusion_matrix.all_reduce()
        num_predictions = self._confusion_matrix.num_predictions
        if not num_predictions:
            raise EmptyMetricError(f"{self.__class__.__qualname__} is empty")

        logger.info(f"Confusion matrix\n{str_full_tensor(self._confusion_matrix.matrix)}\n")
        results = self._confusion_matrix.compute_all_metrics(
            self._betas, prefix=f"{self._metric_name}_"
        )
        results[f"{self._metric_name}_num_predictions"] = num_predictions
        for name, value in results.items():
            logger.info(f"{name}: {str_scalar(value)}")

        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
            engine.create_artifact(
                PyTorchArtifact(
                    tag=f"metric/{self._metric_name}",
                    data={"confusion_matrix": self._confusion_matrix.matrix, "results": results},
                )
            )
        return results


class CategoricalConfusionMatrix(BaseEpochMetric):
    r"""Implements the confusion matrix metric for categorical labels.

    Args:
    ----
        mode (str): Specifies the mode (e.g. train or eval).
        num_classes (int): Specifies the number of classes.
        name (str, optional): Specifies the name of the metric.
            The name is used to log the metric results.
            Default: ``'cat_conf_mat'``
        betas (sequence, optional): Specifies the betas used to
            compute the f-beta score. Default: ``(1,)``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import CategoricalConfusionMatrix
        >>> metric = CategoricalConfusionMatrix("eval", num_classes=3)
        >>> metric
        CategoricalConfusionMatrix(
          (mode): eval
          (name): cat_conf_mat
          (confusion_matrix): MulticlassConfusionMatrix(num_classes=3, num_predictions=0)
          (prediction_transform): ToCategoricalLabel()
        )
        >>> metric(
        ...     torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2]]),
        ...     torch.tensor([0, 1, 0, 1]),
        ... )
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_conf_mat_accuracy': 1.0,
         'eval/cat_conf_mat_balanced_accuracy': 0.666666...,
         'eval/cat_conf_mat_macro_precision': 0.666666...,
         'eval/cat_conf_mat_macro_recall': 0.666666...,
         'eval/cat_conf_mat_macro_f1_score': nan,
         'eval/cat_conf_mat_micro_precision': 1.0,
         'eval/cat_conf_mat_micro_recall': 1.0,
         'eval/cat_conf_mat_micro_f1_score': 1.0,
         'eval/cat_conf_mat_weighted_precision': 1.0,
         'eval/cat_conf_mat_weighted_recall': 1.0,
         'eval/cat_conf_mat_weighted_f1_score': nan,
         'eval/cat_conf_mat_num_predictions': 4}
        >>> metric(torch.tensor([[1, 2, 3], [3, 2, 1]]), torch.tensor([2, 0]))
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_conf_mat_accuracy': 1.0,
         'eval/cat_conf_mat_balanced_accuracy': 1.0,
         'eval/cat_conf_mat_macro_precision': 1.0,
         'eval/cat_conf_mat_macro_recall': 1.0,
         'eval/cat_conf_mat_macro_f1_score': 1.0,
         'eval/cat_conf_mat_micro_precision': 1.0,
         'eval/cat_conf_mat_micro_recall': 1.0,
         'eval/cat_conf_mat_micro_f1_score': 1.0,
         'eval/cat_conf_mat_weighted_precision': 1.0,
         'eval/cat_conf_mat_weighted_recall': 1.0,
         'eval/cat_conf_mat_weighted_f1_score': 1.0,
         'eval/cat_conf_mat_num_predictions': 6}
        >>> metric.reset()
        >>> metric(torch.tensor([[1, 2, 3], [3, 2, 1]]), torch.tensor([2, 0]))
        >>> metric.value()  # doctest:+ELLIPSIS
        {'eval/cat_conf_mat_accuracy': 1.0,
         'eval/cat_conf_mat_balanced_accuracy': 0.666666...,
         'eval/cat_conf_mat_macro_precision': 0.666666...,
         'eval/cat_conf_mat_macro_recall': 0.666666...,
         'eval/cat_conf_mat_macro_f1_score': nan,
         'eval/cat_conf_mat_micro_precision': 1.0,
         'eval/cat_conf_mat_micro_recall': 1.0,
         'eval/cat_conf_mat_micro_f1_score': 1.0,
         'eval/cat_conf_mat_weighted_precision': 1.0,
         'eval/cat_conf_mat_weighted_recall': 1.0,
         'eval/cat_conf_mat_weighted_f1_score': nan,
         'eval/cat_conf_mat_num_predictions': 2}
    """

    def __init__(
        self,
        mode: str,
        num_classes: int,
        name: str = "cat_conf_mat",
        betas: Sequence[int | float] = (1,),
    ) -> None:
        super().__init__(mode=mode, name=name)
        self.prediction_transform = ToCategoricalLabel()

        self._confusion_matrix = CategoricalConfusionMatrixMeter.from_num_classes(num_classes)
        self._betas = tuple(betas)
        self.reset()

    def extra_repr(self) -> str:
        return str_mapping(
            {
                "mode": self._mode,
                "name": self._name,
                "confusion_matrix": self._confusion_matrix,
            }
        )

    def attach(self, engine: BaseEngine) -> None:
        super().attach(engine)
        trackers = [
            MaxScalarHistory(name=f"{self._metric_name}_accuracy"),
            MaxScalarHistory(name=f"{self._metric_name}_balanced_accuracy"),
            MaxScalarHistory(name=f"{self._metric_name}_macro_precision"),
            MaxScalarHistory(name=f"{self._metric_name}_macro_recall"),
            MaxScalarHistory(name=f"{self._metric_name}_micro_precision"),
            MaxScalarHistory(name=f"{self._metric_name}_micro_recall"),
            MaxScalarHistory(name=f"{self._metric_name}_weighted_precision"),
            MaxScalarHistory(name=f"{self._metric_name}_weighted_recall"),
        ]
        for beta in self._betas:
            trackers.append(MaxScalarHistory(name=f"{self._metric_name}_macro_f{beta}_score"))
            trackers.append(MaxScalarHistory(name=f"{self._metric_name}_micro_f{beta}_score"))
            trackers.append(MaxScalarHistory(name=f"{self._metric_name}_weighted_f{beta}_score"))
        for tracker in trackers:
            engine.add_history(tracker)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the mean absolute error metric given a mini-batch of
        examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_clasees)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` and type long or float):
                Specifies the target tensor. The values have to be in
                ``{0, 1, ..., num_classes-1}``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import CategoricalConfusionMatrix
            >>> metric = CategoricalConfusionMatrix("eval", num_classes=3)
            >>> metric(
            ...     torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2]]),
            ...     torch.tensor([0, 1, 0, 1]),
            ... )
            >>> metric.value()  # doctest:+ELLIPSIS
            {'eval/cat_conf_mat_accuracy': 1.0,
             'eval/cat_conf_mat_balanced_accuracy': 0.666666...,
             'eval/cat_conf_mat_macro_precision': 0.666666...,
             'eval/cat_conf_mat_macro_recall': 0.666666...,
             'eval/cat_conf_mat_macro_f1_score': nan,
             'eval/cat_conf_mat_micro_precision': 1.0,
             'eval/cat_conf_mat_micro_recall': 1.0,
             'eval/cat_conf_mat_micro_f1_score': 1.0,
             'eval/cat_conf_mat_weighted_precision': 1.0,
             'eval/cat_conf_mat_weighted_recall': 1.0,
             'eval/cat_conf_mat_weighted_f1_score': nan,
             'eval/cat_conf_mat_num_predictions': 4}
        """
        self._confusion_matrix.update(
            self.prediction_transform(prediction).flatten(), target.flatten()
        )

    def reset(self) -> None:
        self._confusion_matrix.reset()

    def value(self, engine: BaseEngine | None = None) -> dict:
        self._confusion_matrix.all_reduce()
        num_predictions = self._confusion_matrix.num_predictions
        if not num_predictions:
            raise EmptyMetricError(f"{self.__class__.__qualname__} is empty")

        logger.info(f"Confusion matrix\n{str_full_tensor(self._confusion_matrix.matrix)}\n")
        results = self._confusion_matrix.compute_scalar_metrics(
            self._betas, prefix=f"{self._metric_name}_"
        )
        results[f"{self._metric_name}_num_predictions"] = num_predictions
        for name, value in results.items():
            logger.info(f"{name}: {str_scalar(value)}")

        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
            engine.create_artifact(
                PyTorchArtifact(
                    tag=f"metric/{self._metric_name}",
                    data={
                        "confusion_matrix": self._confusion_matrix.matrix,
                        "results": results,
                        "per_class_metrics": self._confusion_matrix.compute_per_class_metrics(),
                    },
                )
            )
        return results
