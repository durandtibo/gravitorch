r"""This module contains implementation to use PyTorch criterion on sequences
with a warm-up approach."""

__all__ = ["WarmupSequenceLoss"]

from typing import Union

from torch import Tensor
from torch.nn import Module

from gravitorch.nn import setup_module


class WarmupSequenceLoss(Module):
    r"""Implements a wrapper to adapt PyTorch loss on sequences with a warm-up
    approach.

    The loss is computed only on the time steps after the warm-up to
    give the model time to warm-up its internal state before making a
    prediction. If ``warmup=5``, the loss is not computed on the first
    five time steps. This loss function assumes that all the sequences
    have the same length.

    This wrapper assumes the loss function has the following two
    inputs:

        - a tensor of prediction of shape
            ``(sequence_length, batch_size, *)`` if
            ``batch_size=False`` or
            ``(batch_size, sequence_length, *)`` otherwise.
        - a tensor of target of shape
            ``(sequence_length, batch_size, *)`` if
            ``batch_size=False`` or
            ``(batch_size, sequence_length, *)`` otherwise.

    Internally, this loss function converts the batch of sequences
    as a batch of examples i.e. the sequence and batch dimensions are
    merged together. Most of the PyTorch criteria work on a batch
    examples i.e. tensors of dimension ``(batch_size, *)`` where ``*``
    means, any number of additional dimensions.

    Note: this is an experimental loss function.

    Args:
        criterion (``torch.nn.Module`` or dict): Specifies the loss
            function/criterion or its configuration.
        warmup (int, optional): Specifies the number of warmup steps.
            Note that this value should be compatible with
            input dimension. Default: ``0``
        batch_first (bool, optional): Indicates if the first dimension
            is the batch or the sequence. If ``True``, the input
            sequence should have the shape
            ``(batch_size, sequence_length, *)``, otherwise
            ``(sequence_length, batch_size, *)``. Default: ``False``
    """

    def __init__(
        self,
        criterion: Union[Module, dict],
        warmup: int = 0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.criterion = setup_module(criterion)
        self._warmup = int(warmup)
        self._batch_first = bool(batch_first)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""Computes the criterion function given the network output and the
        batch.

        Args:
            prediction (``torch.Tensor`` of shape
                ``(sequence_length, batch_size, *)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, *)`` otherwise):
                Specifies the predictions. The tensor type depends on
                the loss function used.
            target (``torch.Tensor`` of shape
                ``(sequence_length, batch_size, *)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, *)`` otherwise):
                Specifies the targets. The tensor type depends on the
                loss function used.

        Returns:
            ``torch.Tensor``: The computed loss value. The shape of
                the tensor depends on the loss function used.
        """
        # Remove the warm-up steps from the prediction and target
        if self._batch_first:
            prediction = prediction[:, self._warmup :]
            target = target[:, self._warmup :]
        else:
            prediction = prediction[self._warmup :, :]
            target = target[self._warmup :, :]
        return self.criterion(prediction.flatten(0, 1), target.flatten(0, 1))
