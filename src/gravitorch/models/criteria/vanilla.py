r"""This module defines a criterion wrapper to make PyTorch criteria compatible
with ``gravitorch.models.VanillaModel``."""

__all__ = ["VanillaLoss"]

from typing import Union

from torch import Tensor
from torch.nn import Module

from gravitorch import constants as ct
from gravitorch.nn import setup_module


class VanillaLoss(Module):
    r"""Implements a wrapper to make compatible most of the PyTorch loss
    functions (a.k.a. criterion) with ``gravitorch.models.VanillaModel``.

    This wrapper assumes the loss function has two inputs:

        - a tensor of prediction.
        - a tensor of target

    The shape and type of the tensors depend on the loss function used.

    Args:
        criterion (``torch.nn.Module`` or dict): Specifies the loss
            function (a.k.a. criterion) or its configuration.
        prediction_key (str): Specifies the prediction key.
            Default: ``"prediction"``.
        target_key (str): Specifies the target key.
            Default: ``"target"``.

    Example usage:

    .. code-block:: python

        >>> from torch import nn
        >>> from gravitorch.models.criteria import VanillaLoss
        # Initialization with a nn.Module
        >>> criterion = VanillaLoss(criterion=nn.MSELoss())
        # Initialization with a config
        >>> criterion = VanillaLoss(criterion={'_target_': 'torch.nn.MSELoss'})
        # Customize keys.
        >>> criterion = VanillaLoss(
        ...     criterion=nn.MSELoss(),
        ...     prediction_key='my_prediction',
        ...     target_key='my_target',
        ... )
        >>> net_out = {'my_prediction': ...}
        >>> batch = {'my_target': ...}
        >>> loss = criterion(net_out, batch)
        {'loss': torch.tensor(...)}
    """

    def __init__(
        self,
        criterion: Union[Module, dict],
        prediction_key: str = ct.PREDICTION,
        target_key: str = ct.TARGET,
    ):
        super().__init__()
        self.criterion = setup_module(criterion)
        self._prediction_key = prediction_key
        self._target_key = target_key

    def forward(self, net_out: dict, batch: dict) -> dict[str, Tensor]:
        r"""Computes the loss value given the network output and the batch.

        Args:
            net_out (dict): Specifies the network output
                which contains the prediction.
            batch (dict): Specifies the batch which contains the target.

        Returns:
            dict: a dict with the loss value.
        """
        prediction = self._get_prediction_from_net_out(net_out)
        target = self._get_target_from_batch(batch)
        return {ct.LOSS: self.criterion(prediction, target)}

    def _get_prediction_from_net_out(self, net_out: dict) -> Tensor:
        r"""Gets the prediction from the network output.

        Args:
            net_out (dict): Specifies the network output
                which contains the prediction.

        Returns:
            ``torch.Tensor``: The prediction
        """
        return net_out[self._prediction_key]

    def _get_target_from_batch(self, batch: dict) -> Tensor:
        r"""Gets the target from the batch. The target is the tensor with the
        key 'target'.

        Args:
            batch (dict): Specifies the batch which contains the target.

        Returns:
            ``torch.Tensor``: The target
        """
        return batch[self._target_key]
