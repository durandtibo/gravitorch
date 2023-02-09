r"""This module defines some accuracy metrics."""

__all__ = ["BinaryAccuracy", "CategoricalAccuracy", "TopKAccuracy"]

import logging
from collections.abc import Sequence
from typing import Optional, Union

from objectory import OBJECT_TARGET
from torch import Tensor
from torch.nn import Identity

from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base_epoch import BaseEpochMetric, BaseStateEpochMetric
from gravitorch.models.metrics.state import AccuracyState, BaseState, setup_state
from gravitorch.nn import ToBinaryLabel, ToCategoricalLabel
from gravitorch.utils.exp_trackers import EpochStep
from gravitorch.utils.format import str_indent, to_torch_mapping_str

logger = logging.getLogger(__name__)


class BinaryAccuracy(BaseStateEpochMetric):
    r"""Implements the binary accuracy metric.

    Args:
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
    """

    def __init__(
        self,
        mode: str,
        threshold: Optional[float] = None,
        name: str = "bin_acc",
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or AccuracyState())
        self.prediction_transform = Identity() if threshold is None else ToBinaryLabel(threshold)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the binary accuracy metric given a mini-batch of examples.

        Args:
            prediction (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type bool or long or float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type bool or long or float):
                Specifies the targets. The values have to be ``0`` or
                ``1``.
        """
        prediction = self.prediction_transform(prediction)
        self._state.update(prediction.eq(target.view_as(prediction)))


class CategoricalAccuracy(BaseStateEpochMetric):
    r"""Implements a categorical accuracy metric.

    Args:
        mode (str): Specifies the mode.
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'cat_acc'``
        state (``BaseState`` or dict or ``None``, optional): Specifies
            the metric state or its configuration. If ``None``,
            ``AccuracyState`` is instantiated. Default: ``None``
    """

    def __init__(
        self,
        mode: str,
        name: str = "cat_acc",
        state: Union[BaseState, dict, None] = None,
    ):
        super().__init__(mode=mode, name=name, state=state or AccuracyState())
        self.prediction_transform = ToCategoricalLabel()

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the accuracy metric given a mini-batch of examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)`` and
                type long or float): Specifies the categorical
                targets. The values have to be in
                ``{0, 1, ..., num_classes-1}``.
        """
        prediction = self.prediction_transform(prediction)
        self._state.update(prediction.eq(target.view_as(prediction)))


class TopKAccuracy(BaseEpochMetric):
    r"""Implements the accuracy at k metric a.k.a. top-k accuracy.

    Args:
        mode (str): Specifies the mode.
        topk (list or tuple, optional): Specifies the k values used to
            evaluate the top-k accuracy metric. Default: ``(1, 5)``
        name (str, optional): Specifies the name used to log the
            metric. Default: ``'accuracy'``
    """

    def __init__(
        self,
        mode: str,
        topk: Sequence[int] = (1, 5),
        name: str = "acc_top",
        state_config: Optional[dict] = None,
    ):
        super().__init__(mode, name)
        self._topk = topk if isinstance(topk, tuple) else tuple(topk)
        self._maxk = max(self._topk)

        self._states: dict[int, BaseState] = {
            tol: setup_state(
                state_config or {OBJECT_TARGET: "gravitorch.models.metrics.state.AccuracyState"}
            )
            for tol in self._topk
        }

    def extra_repr(self) -> str:
        return (
            f"mode={self._mode},\nname={self._name},\ntopk={self._topk},\n"
            f"states:\n  {str_indent(to_torch_mapping_str(self._states))}"
        )

    @property
    def topk(self) -> tuple[int, ...]:
        return self._topk

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to the provided engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        super().attach(engine)
        for k, state in self._states.items():
            for history in state.get_histories(f"{self._metric_name}_{k}_"):
                engine.add_history(history)

    def forward(self, prediction: Tensor, target: Tensor) -> None:
        r"""Updates the accuracy metric given a mini-batch of examples.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the predictions.
            target (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn)`` or ``(d0, d1, ..., dn, 1)``
                and type long or float): Specifies the targets.
                The values have to be in
                ``{0, 1, ..., num_classes-1}``.
        """
        _, pred = prediction.topk(self._maxk, -1, True, True)
        correct = pred.eq(target.view(*pred.shape[:-1], 1).expand_as(pred)).float()
        for k, state in self._states.items():
            state.update(correct[..., :k].sum(dim=-1))

    def reset(self) -> None:
        r"""Resets the metric."""
        for state in self._states.values():
            state.reset()

    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates the metric and log the results given all the examples
        previously seen.

        Args:
            engine (``BaseEngine``, optional): Specifies the engine.
                This argument is required to log the results.
                Default: ``None``.

        Returns:
             dict: The results of the metric
        """
        results = {}
        for k, state in self._states.items():
            results.update(state.value(f"{self._metric_name}_{k}_"))
        if engine:
            engine.log_metrics(results, EpochStep(engine.epoch))
        return results
