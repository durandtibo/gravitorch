r"""This module defines a criterion wrapper to make PyTorch criteria compatible
with ``gravitorch.models.VanillaModel`` and packed sequences."""

__all__ = ["PackedSequenceLoss"]

from typing import Union

from torch import Tensor
from torch.nn import Module

from gravitorch import constants as ct
from gravitorch.nn import setup_module


class PackedSequenceLoss(Module):
    r"""Implements a wrapper to adapt PyTorch criterion to deal with packed
    sequence inputs.

    This class works for every PyTorch criterion that has two inputs:
    prediction and target.

    Args:
        criterion (``torch.nn.Module`` or dict): Specifies the loss
            function or its configuration.
        prediction_key (str): Specifies the prediction key.
            Default: ``"prediction"``.
        target_key (str): Specifies the target key.
            Default: ``"target"``.
        mask_key (str): Specifies the mask key.
            Default: ``"mask"``.

    Example usage:

    .. code-block:: python

        >>> from torch import nn
        >>> from gravitorch.models.criteria import PackedSequenceLoss
        # Init with a nn.Module
        >>> criterion = PackedSequenceLoss(criterion=nn.MSELoss())
        # Init with a config
        >>> criterion = PackedSequenceLoss(criterion={'_target_': 'torch.nn.MSELoss'})
        # Customize keys.
        >>> criterion = PackedSequenceLoss(
        ...     criterion=nn.MSELoss(),
        ...     prediction_key='my_prediction',
        ...     target_key='my_target',
        ...     mask_key='my_mask',
        ... )
        >>> net_out = {'my_prediction': ...}
        >>> batch = {'my_target': ..., 'my_mask': ...}
        >>> loss = criterion(net_out, batch)
        {'loss': torch.tensor(...)}
    """

    def __init__(
        self,
        criterion: Union[Module, dict],
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
        mask_key: str = ct.MASK,
    ):
        super().__init__()
        self.criterion = setup_module(criterion)
        self._prediction_key = prediction_key
        self._target_key = target_key
        self._mask_key = mask_key

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Computes the loss value given the network output and the batch.

        Args:
            net_out (dict): Specifies the network output which
                contains the prediction.
            batch (dict): Specifies the batch which contains the
                target and the mask.

        Returns:
            dict: a dict with the loss value.
        """
        prediction = self._get_prediction_from_net_out(net_out)
        target = self._get_target_from_batch(batch)

        # Get the mask and remove the examples that are 0-masked.
        mask = batch.get(self._mask_key, None)
        if mask is not None:
            prediction = prediction[mask.data > 0]
            target = target[mask.data > 0]

        return {ct.LOSS: self.criterion(prediction, target)}

    def _get_prediction_from_net_out(self, net_out: dict) -> Tensor:
        r"""Gets the prediction from the network output.

        Args:
            net_out (dict): Specifies the network output
                which contains the prediction.

        Returns:
            ``torch.Tensor``: the prediction
        """
        return net_out[self._prediction_key].data

    def _get_target_from_batch(self, batch: dict) -> Tensor:
        r"""Gets the target from the batch. The target is the tensor with the
        key 'target'.

        Args:
            batch (dict): Specifies the batch which contains the target.

        Returns:
            ``torch.Tensor``: the target
        """
        return batch[self._target_key].data
