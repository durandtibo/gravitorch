r"""This module contains some layers to shift and scale a sequence."""

__all__ = ["ShiftScale", "SequenceShiftScale", "shift_scale", "sequence_shift_scale"]

from torch import Tensor
from torch.nn import Module


class ShiftScale(Module):
    r"""Implements a layer to shift and scale an input sequence from a source
    range to a destination range."""

    def forward(self, tensor: Tensor, src_range: Tensor, dst_range: Tensor) -> Tensor:
        r"""Shifts and scales the input sequence from a source range to a
        destination range.

        Args:
            tensor (``torch.Tensor`` of type float and shape
                ``(batch_size, feature_size)``): Specifies the input
                sequence.
            src_range (``torch.Tensor`` of type float and shape
                ``(batch_size, 2, feature_size)``): Specifies the min
                and max values of the source space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and
                max values should be different.
            dst_range (``torch.Tensor`` of type float and shape
            ``(batch_size, 2, feature_size)``): Specifies the min
                and max values of the destination space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and max
                values should be different.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(batch_size, feature_size)``: The shifted and scaled
                sequence.
        """
        return shift_scale(tensor, src_range, dst_range)


class SequenceShiftScale(Module):
    r"""Implements a layer to shift and scale sequences from a source range to a
    destination range.

    Args:
        batch_first (bool, optional): Indicates if the first
            dimension is the batch or the sequence. If ``True``, the
            input sequence should have the shape
            ``(batch_size, sequence_length, feature_size)``, otherwise
            ``(sequence_length, batch_size, feature_size)``.
            Default: ``False``
    """

    def __init__(self, batch_first: bool = False):
        super().__init__()
        self._batch_first = bool(batch_first)

    @property
    def batch_first(self) -> bool:
        r"""bool: Indicates if the first dimension is the batch or
        the sequence. If ``True``, the input representation
        should have the shape
        ``(batch_size, sequence_length, feature_size)``, otherwise
        ``(sequence_length, batch_size, feature_size)``.
        """
        return self._batch_first

    def extra_repr(self) -> str:
        return f"batch_first={self._batch_first}"

    def forward(self, sequence: Tensor, src_range: Tensor, dst_range: Tensor) -> Tensor:
        r"""Shifts and scales a sequence from a source range to a destination
        range.

        Args:
            sequence (``torch.Tensor`` of type float and shape
                ``(batch_size, sequence_length, feature_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, feature_size)``
                otherwise): Specifies the sequence to shift and scale.
            src_range (``torch.Tensor`` of type float and shape
                ``(batch_size, 2, feature_size)``): Specifies the min
                and max values of the source space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and max
                values should be different.
            dst_range (``torch.Tensor`` of type float and shape
                ``(batch_size, 2, feature_size)``): Specifies the min
                and max values of the destination space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and max
                values should be different.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(sequence_length, batch_size, feature_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, feature_size)``
                otherwise: The shifted and scaled sequence.
        """
        return sequence_shift_scale(sequence, src_range, dst_range, self._batch_first)


def shift_scale(tensor: Tensor, src_range: Tensor, dst_range: Tensor) -> Tensor:
    r"""Shifts and scales a tensor from a source range to a destination range.

    Args:
        tensor (``torch.Tensor`` of type float and shape
            ``(batch_size, feature_size)``): Specifies the tensor to
            shift and scale.
        src_range (``torch.Tensor`` of type float and shape
            ``(batch_size, 2, feature_size)``): Specifies the min and
            max values of the source space. ``[:, 0, :]`` indicate
            the min values and ``[:, 1, :]`` indicates the max values.
            For a given dimension, the min and max values should be
            different.
        dst_range (``torch.Tensor`` of type float and shape
        ``(batch_size, 2, feature_size)``): Specifies the min and
            max values of the destination space. ``[:, 0, :]``
            indicates the min values and ``[:, 1, :]`` indicates the
            max values. For a given dimension, the min and max values
            should be different.

    Returns:
        ``torch.Tensor`` of type float and shape
            ``(batch_size, feature_size)``: The shifted and scaled
            tensor.
    """
    src_diff = src_range[:, 1] - src_range[:, 0]
    dst_diff = dst_range[:, 1] - dst_range[:, 0]
    scale = dst_diff / src_diff
    return (tensor - src_range[:, 0]) * scale + dst_range[:, 0]


def sequence_shift_scale(
    sequence: Tensor,
    src_range: Tensor,
    dst_range: Tensor,
    batch_first: bool = False,
) -> Tensor:
    r"""Shifts and scales a sequences from a source range to a destination
    range.

    Args:
        sequence (``torch.Tensor`` of type float and shape
            ``(batch_size, sequence_length, feature_size)`` if
            ``batch_size=False`` or
            ``(batch_size, sequence_length, feature_size)``
            otherwise): Specifies the sequence to shift and scale.
        src_range (``torch.Tensor`` of type float and shape
            ``(batch_size, 2, feature_size)``): Specifies the min and
            max values of the source space. ``[:, 0, :]`` indicates
            the min values and ``[:, 1, :]`` indicates the max values.
            For a given dimension, the min and max values should be
            different.
        dst_range (``torch.Tensor`` of type float and shape
            ``(batch_size, 2, feature_size)``): Specifies the min and
            max values of the destination space. ``[:, 0, :]``
            indicates the min values and ``[:, 1, :]`` indicates the
            max values. For a given dimension, the min and max values
            should be different.
        batch_first (bool, optional): Indicates if the first dimension
            is the batch or the sequence. If ``True``, the input
            sequence should have the shape
            ``(batch_size, sequence_length, feature_size)``, otherwise
            ``(sequence_length, batch_size, feature_size)``.
            Default: ``False``

    Returns:
        ``torch.Tensor`` of type float and shape
            ``(sequence_length, batch_size, feature_size)``
            if ``batch_size=False`` or
            ``(batch_size, sequence_length, feature_size)`` otherwise:
            The shifted and scaled tensor.
    """
    src_diff = src_range[:, 1] - src_range[:, 0]
    dst_diff = dst_range[:, 1] - dst_range[:, 0]
    scale = dst_diff / src_diff
    if batch_first:
        return (sequence - src_range[:, None, 0]) * scale[:, None, :] + dst_range[:, None, 0]
    return (sequence - src_range[:, 0]) * scale + dst_range[:, 0]
