r"""This module defines some accuracy metrics."""

from __future__ import annotations

__all__ = ["BinaryAccuracy", "CategoricalAccuracy", "TopKAccuracy"]

import logging
from collections.abc import Sequence

from coola.utils import str_mapping
from objectory import OBJECT_TARGET
from torch import Tensor
from torch.nn import Identity

from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base_epoch import BaseEpochMetric, BaseStateEpochMetric
from gravitorch.models.metrics.state import AccuracyState, BaseState, setup_state
from gravitorch.nn import ToBinaryLabel, ToCategoricalLabel
from gravitorch.utils.exp_trackers import EpochStep

logger = logging.getLogger(__name__)


class BinaryAccuracy(BaseStateEpochMetric):
    r"""Implements the binary accuracy metric.

    Args:
    ----
        mode (str): Specifies the mode.
        threshold (float or ``None``, optional): Specifies a threshold
            value to generate the predicted labels from the
            predictions. If ``None``, the predictions are interpreted
            as the predicted labels. Default: ``None``
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'bin_acc'``
        state (``BaseState`` or dict or ``None``, optional): Specifies
            the metric state or its configuration. If ``None``,
            ``AccuracyState`` is instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import BinaryAccuracy
        >>> metric = BinaryAccuracy("eval")
        >>> metric
        BinaryAccuracy(
          (mode): eval
          (name): bin_acc
          (state): AccuracyState(num_predictions=0)
          (prediction_transform): Identity()
        )
        >>> metric(torch.zeros(4), torch.ones(4))
        >>> metric.value()
        {'eval/bin_acc_accuracy': 0.0, 'eval/bin_acc_num_predictions': 4}
        >>> metric(torch.ones(4), torch.ones(4))
        >>> metric.value()
        {'eval/bin_acc_accuracy': 0.5, 'eval/bin_acc_num_predictions': 8}
        >>> metric.reset()
        >>> metric(torch.ones(4), torch.ones(4))
        >>> metric.value()
        {'eval/bin_acc_accuracy': 1.0, 'eval/bin_acc_num_predictions': 4}
    """

    def __init__(
        self,
        mode: str,
        threshold: float | None = None,
        name: str = "bin_acc",
        state: BaseState | dict | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name, state=state or AccuracyState())
        self.prediction_transform = Identity() if threshold is None else ToBinaryLabel(threshold)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the binary accuracy metric given a mini-batch of
        examples.

        Args:
        ----
            prediction (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type bool or long or float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type bool or long or float):
                Specifies the targets. The values have to be ``0`` or
                ``1``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import BinaryAccuracy
            >>> metric = BinaryAccuracy("eval")
            >>> metric(torch.zeros(4), torch.ones(4))
            >>> metric.value()
            {'eval/bin_acc_accuracy': 0.0, 'eval/bin_acc_num_predictions': 4}
        """
        prediction = self.prediction_transform(prediction)
        self._state.update(prediction.eq(target.view_as(prediction)))


class CategoricalAccuracy(BaseStateEpochMetric):
    r"""Implements a categorical accuracy metric.

    Args:
    ----
        mode (str): Specifies the mode.
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'cat_acc'``
        state (``BaseState`` or dict or ``None``, optional): Specifies
            the metric state or its configuration. If ``None``,
            ``AccuracyState`` is instantiated. Default: ``None``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import CategoricalAccuracy
        >>> metric = CategoricalAccuracy("eval")
        >>> metric
        CategoricalAccuracy(
          (mode): eval
          (name): cat_acc
          (state): AccuracyState(num_predictions=0)
          (prediction_transform): ToCategoricalLabel()
        )
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 0]))
        >>> metric.value()
        {'eval/cat_acc_accuracy': 1.0, 'eval/cat_acc_num_predictions': 2}
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 2]))
        >>> metric.value()
        {'eval/cat_acc_accuracy': 0.75, 'eval/cat_acc_num_predictions': 4}
        >>> metric.reset()
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 2]))
        >>> metric.value()
        {'eval/cat_acc_accuracy': 0.5, 'eval/cat_acc_num_predictions': 2}
    """

    def __init__(
        self,
        mode: str,
        name: str = "cat_acc",
        state: BaseState | dict | None = None,
    ) -> None:
        super().__init__(mode=mode, name=name, state=state or AccuracyState())
        self.prediction_transform = ToCategoricalLabel()

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the accuracy metric given a mini-batch of examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)`` and
                type long or float): Specifies the categorical
                targets. The values have to be in
                ``{0, 1, ..., num_classes-1}``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import CategoricalAccuracy
            >>> metric = CategoricalAccuracy("eval")
            >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 0]))
            >>> metric.value()
            {'eval/cat_acc_accuracy': 1.0, 'eval/cat_acc_num_predictions': 2}
        """
        prediction = self.prediction_transform(prediction)
        self._state.update(prediction.eq(target.view_as(prediction)))


class TopKAccuracy(BaseEpochMetric):
    r"""Implements the accuracy at k metric a.k.a. top-k accuracy.

    Args:
    ----
        mode (str): Specifies the mode.
        topk (list or tuple, optional): Specifies the k values used to
            evaluate the top-k accuracy metric. Default: ``(1, 5)``
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'accuracy'``

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from gravitorch.models.metrics import TopKAccuracy
        >>> metric = TopKAccuracy("eval", topk=(1,))
        >>> metric
        TopKAccuracy(
          (mode): eval
          (name): acc_top
          (topk): (1,)
          (states):
            (1): AccuracyState(num_predictions=0)
        )
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 0]))
        >>> metric.value()
        {'eval/acc_top_1_accuracy': 1.0, 'eval/acc_top_1_num_predictions': 2}
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 2]))
        >>> metric.value()
        {'eval/acc_top_1_accuracy': 0.75, 'eval/acc_top_1_num_predictions': 4}
        >>> metric.reset()
        >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 2]))
        >>> metric.value()
        {'eval/acc_top_1_accuracy': 0.5, 'eval/acc_top_1_num_predictions': 2}
    """

    def __init__(
        self,
        mode: str,
        topk: Sequence[int] = (1, 5),
        name: str = "acc_top",
        state_config: dict | None = None,
    ) -> None:
        super().__init__(mode, name)
        self._topk = topk if isinstance(topk, tuple) else tuple(topk)
        self._maxk = max(self._topk)

        if state_config is None:
            state_config = {OBJECT_TARGET: "gravitorch.models.metrics.state.AccuracyState"}
        self._states: dict[int, BaseState] = {tol: setup_state(state_config) for tol in self._topk}

    def extra_repr(self) -> str:
        return str_mapping(
            {
                "mode": self._mode,
                "name": self._name,
                "topk": self._topk,
                "states": "\n" + str_mapping(self._states),
            }
        )

    @property
    def topk(self) -> tuple[int, ...]:
        return self._topk

    def attach(self, engine: BaseEngine) -> None:
        super().attach(engine)
        for k, state in self._states.items():
            for history in state.get_histories(f"{self._metric_name}_{k}_"):
                engine.add_history(history)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the accuracy metric given a mini-batch of examples.

        Args:
        ----
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type long or float): Specifies the targets.
                The values have to be in
                ``{0, 1, ..., num_classes-1}``.

        Example usage:

        .. code-block:: pycon

            >>> import torch
            >>> from gravitorch.models.metrics import TopKAccuracy
            >>> metric = TopKAccuracy("eval", topk=(1,))
            >>> metric(torch.tensor([[0, 2, 1], [2, 1, 0]]), torch.tensor([1, 0]))
            >>> metric.value()
            {'eval/acc_top_1_accuracy': 1.0, 'eval/acc_top_1_num_predictions': 2}
        """
        _, pred = prediction.topk(self._maxk, -1, True, True)
        correct = pred.eq(target.view(*pred.shape[:-1], 1).expand_as(pred)).float()
        for k, state in self._states.items():
            state.update(correct[..., :k].sum(dim=-1))

    def reset(self) -> None:
        for state in self._states.values():
            state.reset()

    def value(self, engine: BaseEngine | None = None) -> dict:
        results = {}
        for k, state in self._states.items():
            results.update(state.value(f"{self._metric_name}_{k}_"))
        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
        return results
