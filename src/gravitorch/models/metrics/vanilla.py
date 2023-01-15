"""This module defines some classes to make PyTorch metric compatible with
``VanillaModel``."""

__all__ = ["PaddedSequenceMetric", "VanillaMetric"]

from typing import Optional, Union

import torch
from torch import Tensor

from gravitorch import constants as ct
from gravitorch.engines.base import BaseEngine
from gravitorch.models.metrics.base import BaseMetric


class VanillaMetric(BaseMetric):
    r"""Implements a wrapper to make compatible most of the metrics with
    ``gravitorch.models.VanillaModel``.

    This class works for every ``BaseMetric`` instance that has two
    inputs: prediction and target.

    Args:
        metric (``BaseMetric`` or dict): Specifies the metric function
            or its configuration.
        mode (str, optional): Specifies the mode (e.g train or eval).
        prediction_key (str, optional): Specifies the key where the
            prediction is if the network output is a dict.
            Default: ``prediction``
        target_key (str, optional): Specifies the key where the target
            is in the batch input. Default: ``target``

    Example usage:

    .. code-block:: python

        >>> from gravitorch.models.metrics import TopKAccuracy
        # Initialization with a metric object.
        >>> metric = VanillaMetric(TopKAccuracy(mode='train', topk=[1]))
        # Initialization with the config of a metric.
        >>> metric = VanillaMetric(mode='eval', metric={'_target_': 'TopKAccuracy', 'topk': [1]})
        # Customize keys.
        >>> net_out = {'next_sentence_prediction': ...}
        >>> batch = {'next_sentence_target': ...}
        >>> metric = VanillaMetric(
        ...     TopKAccuracy(mode='train', topk=[1]),
        ...     prediction_key='next_sentence_prediction',
        ...     target_key='next_sentence_target',
        ... )
    """

    def __init__(
        self,
        metric: Union[BaseMetric, dict],
        mode: Optional[str] = None,
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
    ):
        super().__init__()
        self.metric = self._setup_metric(metric=metric, mode=mode)
        self._prediction_key = prediction_key
        self._target_key = target_key

    def attach(self, engine: BaseEngine) -> None:
        r"""Attaches current metric to provided engine.

        This method can be used to:

            - add event handler to the engine
            - set up history trackers

        Args:
            engine (``BaseEngine``): Specifies the engine.
        """
        self.metric.attach(engine)

    def forward(self, cri_out: dict, net_out: dict, batch: dict) -> Optional[dict]:
        r"""Updates the metric given a mini-batch of examples.

        Args:
            cri_out (dict): Specifies the criterion output. (not used
                in this implementation)
            net_out (dict or ``torch.Tensor``): Specifies the network
                output which contains the prediction.
            batch (dict): Specifies the batch which contains the
                target.

        Returns:
            dict or ``None``: A dict with the metric values. ``None``
                means no metric value is returned.
        """
        return self.metric(
            self._get_prediction_from_net_out(net_out),
            self._get_target_from_batch(batch),
        )

    def reset(self) -> None:
        r"""Resets the metric."""
        self.metric.reset()

    def value(self, engine: Optional[BaseEngine] = None) -> dict:
        r"""Evaluates the metric and log the results given all the examples
        previously seen.

        Args:
            engine (``BaseEngine`` or None): Specifies the engine.

        Returns:
             dict: The results of the metric.
        """
        return self.metric.value(engine)

    def _get_prediction_from_net_out(self, net_out: dict) -> Tensor:
        r"""Gets the prediction from the network output.

        Args:
            net_out (dict): Specifies the network output which
                contains the prediction.

        Returns:
            ``torch.Tensor``: The tensor of predictions.
        """
        return net_out[self._prediction_key].detach()

    def _get_target_from_batch(self, batch: dict) -> Tensor:
        r"""Gets the target from the batch. The target is the tensor with the
        key 'target'.

        Args:
            batch (dict): Specifies the batch which contains the target.

        Returns:
            ``torch.Tensor``: The tensor of targets.
        """
        return batch[self._target_key].detach()

    def _setup_metric(self, metric: Union[BaseMetric, dict], mode: Optional[str]) -> BaseMetric:
        r"""Sets up the metric.

        If the input is a dict containing the configuration of the
        metric, it will instantiate the metric given its config by
        using the ``factory`` method of ``BaseMetric``.

        Args:
            metric (``BaseMetric`` or dict): Specifies the metric
                function or its configuration.
            mode (str, optional): Specifies the mode (e.g train or
                eval).

        Returns:
            ``BaseMetric``: an instance of the metric.
        """
        if isinstance(metric, dict):
            if mode is not None:
                metric["mode"] = mode
            metric = BaseMetric.factory(**metric)
        return metric


class PaddedSequenceMetric(VanillaMetric):
    r"""Implements a wrapper to make compatible most of the metrics with
    ``gravitorch.models.VanillaModel``.

    This metric is designed to work on sequences. This metric should
    have at least two inputs:

        - the prediction which is a ``torch.Tensor`` of shape
            ``(sequence_length, batch_size, *)`` or
            ``(batch_size, sequence_length, *)`` where ``*`` means
            any number of additional dimensions. This tensor is
            converted to a tensor of shape
            ``(sequence_length * batch_size, *)`` and then feeds to
            the wrapped metric.
        - the target which is a ``torch.Tensor`` of shape
            ``(sequence_length, batch_size, *)`` or
            ``(batch_size, sequence_length, *)`` where ``*`` means
            any number of additional dimensions.
            This tensor is converted to a tensor of shape
            ``(sequence_length * batch_size, *)`` and then feeds to
            the wrapped metric.

    The input mask is optional. If no mask is provided, all the steps
    are considered as valid. The mask is a ``torch.Tensor`` of shape
    ``(sequence_length, batch_size)`` or
    ``(batch_size, sequence_length)``. The type of the tensor can be
    ``torch.int`` or``torch.long`` or``torch.float`` or ``torch.bool``
    with the following values:

        - valid value: ``True`` or ``1`` if ``valid_value=True``,
            otherwise ``False`` or ``0``.
        - invalid value: ``False`` or ``0`` if ``valid_value=True``,
            otherwise ``True`` or ``1``.

    Args:
        metric (``BaseMetric`` or dict): Specifies the metric function
            or its configuration.
        mode (str, optional): Specifies the mode (e.g train or eval).
        prediction_key (str, optional): Specifies the key where the
            prediction is if the network output is a dict.
            Default: ``prediction``
        target_key (str, optional): Specifies the key where the target
            is in the batch input. Default: ``target``
        mask_key (str, optional): Specifies the key where the target
            is in the batch input. Default: ``target``
        valid_value (bool, optional): Indicates the valid values in
            the mask. If ``True``, the valid values are indicated by
            a ``True`` in the mask. If ``False``, the valid values are
            indicated by a ``False`` in the mask. Default: ``True``
        mask_in_batch (bool, optional): Indicates if the mask is in
            ``batch`` or ``net_out``. If ``True``, the mask is taken
            from the input ``batch``, otherwise it is taken from the
            input ``net_out``. Default: ``True``

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.models.metrics import PaddedSequenceMetric, AbsoluteError
        >>> metric = PaddedSequenceMetric(AbsoluteError(mode='train'))
        >>> metric(
        ...     cri_out={},
        ...     net_out={'prediction': torch.ones(2, 6, 1)},
        ...     batch={'target': torch.zeros(2, 6, 1)},
        ... )
        >>> metric.value()
        {'train/mae': 1.0, 'train/mae_num_examples': 12}
        >>> metric(
        ...     cri_out={},
        ...     net_out={'prediction': torch.ones(2, 4, 1)},
        ...     batch={
        ...         'target': torch.zeros(2, 4, 1),
        ...         'mask': torch.tensor([[True, True, False, True], [True, True, True, False]]),
        ...    }
        ... )
        >>> metric.value()
        {'train/mae': 1.0, 'train/mae_num_examples': 18}
    """

    def __init__(
        self,
        metric: Union[BaseMetric, dict],
        mode: Optional[str] = None,
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
        mask_key: str = ct.MASK,
        valid_value: bool = True,
        mask_in_batch: bool = True,
    ):
        super().__init__(
            metric=metric,
            mode=mode,
            prediction_key=prediction_key,
            target_key=target_key,
        )
        self._mask_key = mask_key
        self._valid_value = bool(valid_value)
        self._mask_in_batch = bool(mask_in_batch)

    def forward(self, cri_out: dict, net_out: dict, batch: dict) -> Optional[dict]:
        r"""Updates the metric given a mini-batch of examples.

        Args:
            cri_out (dict): Specifies the criterion output. (not used
                in this implementation)
            net_out (dict): Specifies the network output which
                contains the prediction.
            batch (dict): Specifies the batch which contains the
                target and the mask.

        Returns:
            dict or ``None``: A dict with the metric values. ``None``
                means no metric value is returned.
        """
        prediction = self._get_prediction_from_net_out(net_out)
        target = self._get_target_from_batch(batch)

        # See the batch of sequences as a batch of examples
        prediction = prediction.view(-1, *prediction.shape[2:])
        target = target.view(-1, *target.shape[2:])

        # Get the mask and remove the examples that are masked
        mask = (
            batch.get(self._mask_key, None)
            if self._mask_in_batch
            else net_out.get(self._mask_key, None)
        )
        if mask is not None:
            mask = mask.view(-1).bool()
            if not self._valid_value:
                mask = torch.logical_not(mask)
            prediction = prediction[mask]
            target = target[mask]

        return self.metric(prediction, target)
