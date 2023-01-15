r"""This module implements some random Fourier feature (RFF) layers."""

__all__ = ["SequenceGaussianRFF", "ScaleShiftSequenceGaussianRFF"]

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from gravitorch.nn.shift_scale import SequenceShiftScale
from gravitorch.nn.utils import get_module_device


class SequenceGaussianRFF(Module):
    r"""Implements a Gaussian random Fourier feature (RFF) encoder for
    sequences.

    This encoder was proposed in the following paper:

        Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains.
        Tancik M., Srinivasan PP., Mildenhall B., Fridovich-Keil S.,
        Raghavan N., Singhal U., Ramamoorthi R., Barron JT., Ng R.
        In NeurIPS, 2020. http://arxiv.org/pdf/2006.10739

    Args:
        input_size (int): Specifies the input size.
        output_size (int): Specifies the output size.
        sigma (float, optional): Specifies the standard deviation of
            the Gaussian distribution. Default: ``2*pi``
        trainable_params (bool, optional): If ``True``, the
            parameters sampled from the Gaussian distributed are
            optimized during training. Default: ``False``.
        batch_first (bool, optional): Indicates if the first dimension
            is the batch or the sequence. If ``True``, the input
            sequence should have the shape
            ``(batch_size, sequence_length, input_size)``, otherwise
            ``(sequence_length, batch_size, input_size)``. Note that
            the sub-modules of this layer should be compatible with
            the batch format. Default: ``False``
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        sigma: float = 2 * math.pi,
        trainable_params: bool = False,
        batch_first: bool = False,
    ):
        super().__init__()
        self._input_size = int(input_size)
        self._intermediate_output_size = output_size // 2
        self._output_size = int(output_size)
        self._sigma = float(sigma)
        self._batch_first = bool(batch_first)

        self.gaussian = Parameter(
            torch.randn(input_size, self._intermediate_output_size) * self._sigma,
            requires_grad=trainable_params,
        )

    @property
    def batch_first(self) -> bool:
        r"""bool: Indicates if the first dimension is the batch or
        the sequence.

        If ``True``, the input sequence should have the shape
        ``(batch_size, sequence_length, input_size)``, otherwise
        ``(sequence_length, batch_size, input_size)``.
        """
        return self._batch_first

    @property
    def input_size(self) -> int:
        r"""int: The input size of the sequence."""
        return self._input_size

    def extra_repr(self) -> str:
        return (
            f"input_size={self._input_size}, output_size={self._output_size}, "
            f"sigma={self._sigma}, batch_first={self._batch_first}"
        )

    def forward(self, sequence: Tensor) -> Tensor:
        r"""Computes the Fourier feature mappings of the input representation.

        Args:
            sequence (``torch.Tensor`` of type float and shape
                ``(batch_size, sequence_length, input_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, input_size)``
                otherwise): Specifies the
                input sequence. The values should be in the range
                ``[0, 1)``.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(sequence_length, batch_size, output_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, output_size)``
                otherwise: The computed Fourier feature mappings.
        """
        x_proj = torch.mm((2 * math.pi * sequence).view(-1, self._input_size), self.gaussian).view(
            *sequence.shape[:2], self._intermediate_output_size
        )
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=2)

    def get_dummy_input(self, batch_size: int = 1, seq_len: int = 1) -> tuple[Tensor]:
        r"""Generates a dummy input for this network.

        Args:
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            seq_len (int, optional): Specifies the sequence length.
                Default: ``1``

        Returns:
            A tuple of one torch tensor:
                - The sequence of continuous inputs: ``torch.Tensor``
                    of type float and shape
                    ``(sequence_length, batch_size, input_size)`` if
                    ``batch_first=False`` or
                    ``(batch_size, sequence_length, input_size)``
                    otherwise.
        """
        device = self.gaussian.data.device
        if self._batch_first:
            return (torch.rand(batch_size, seq_len, self._input_size, device=device),)
        return (torch.rand(seq_len, batch_size, self._input_size, device=device),)


class ScaleShiftSequenceGaussianRFF(Module):
    r"""Implements a Gaussian random Fourier feature (RFF) encoder for
    sequences, where the values are shifted and scaled.

    Unlike ``SequenceGaussianRFF``, this layer can deal with input
    values that are not in the range ``[0, 1)``.

    This encoder was proposed in the following paper:

        Fourier Features Let Networks Learn High Frequency Functions
        in Low Dimensional Domains.
        Tancik M., Srinivasan PP., Mildenhall B., Fridovich-Keil S.,
        Raghavan N., Singhal U., Ramamoorthi R., Barron JT., Ng R.
        In NeurIPS, 2020. http://arxiv.org/pdf/2006.10739

    Args:
        input_size (int): Specifies the input size.
        output_size (int): Specifies the output size.
        sigma (float, optional): Specifies the standard deviation of
            the Gaussian distribution. Default: ``2*pi``
        trainable_params (bool, optional): If ``True``, the parameters
            sampled from the Gaussian distributed are optimized during
            training. Default: ``False``.
        batch_first (bool, optional): Indicates if the first dimension
            is the batch or the sequence. If ``True``,
            the input sequence should have the shape
            ``(batch_size, sequence_length, input_size)``, otherwise
            ``(sequence_length, batch_size, input_size)``. Note that
            the sub-modules of this layer should be compatible with
            the batch format. Default: ``False``
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        sigma: float = 2 * math.pi,
        trainable_params: bool = False,
        batch_first: bool = False,
    ):
        super().__init__()
        self._input_size = input_size
        self._batch_first = batch_first

        self.shift_scale = SequenceShiftScale(batch_first=self._batch_first)
        self.rff = SequenceGaussianRFF(
            input_size=input_size,
            output_size=output_size,
            sigma=sigma,
            trainable_params=trainable_params,
            batch_first=batch_first,
        )

    @property
    def batch_first(self) -> bool:
        r"""bool: Indicates if the first dimension is the batch or the
        sequence. If ``True``, the input representation should have
        the shape ``(batch_size, sequence_length, input_size)``,
        otherwise ``(sequence_length, batch_size, input_size)``.
        """
        return self._batch_first

    def forward(self, sequence: Tensor, src_range: Tensor, dst_range: Tensor) -> Tensor:
        r"""Computes the Fourier feature mappings of the input representation.

        Args:
            sequence (``torch.Tensor`` of type float and shape
                ``(batch_size, sequence_length, input_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, input_size)``
                otherwise): Specifies the input sequence.
            src_range (``torch.Tensor`` of type float and shape
                ``(batch_size, 2, input_size)``): Specifies the min
                and max values of the source space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and
                max values should be different.
            dst_range (``torch.Tensor`` of type float and shape
                ``(batch_size, 2, input_size)``): Specifies the min
                and max values of the destination space. ``[:, 0, :]``
                indicates the min values and ``[:, 1, :]`` indicates
                the max values. For a given dimension, the min and
                max values should be different.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(sequence_length, batch_size, output_size)`` if
                ``batch_size=False`` or
                ``(batch_size, sequence_length, output_size)``
                otherwise: The computed Fourier feature mappings.
        """
        return self.rff(self.shift_scale(sequence, src_range, dst_range))

    def get_dummy_input(
        self, batch_size: int = 1, seq_len: int = 1
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Generates a dummy input for this network.

        Args:
            batch_size (int, optional): Specifies the batch size.
                Default: ``1``
            seq_len (int, optional): Specifies the sequence length.
                Default: ``1``

        Returns:
            A tuple of three torch tensors:
                - The sequence of continuous inputs: ``torch.Tensor``
                    of type float and shape
                    ``(sequence_length, batch_size, input_size)`` if
                    ``batch_first=False`` or
                    ``(batch_size, sequence_length, input_size)``
                    otherwise.
                - The min and max values of the source space:
                    ``torch.Tensor`` of type float and shape
                    ``(batch_size, 2, input_size)``
                - The min and max values of the destination space:
                    ``torch.Tensor`` of type float and shape
                    ``(batch_size, 2, input_size)``
        """
        device = get_module_device(self.rff)
        src_range = torch.ones(batch_size, 2, self._input_size, device=device)
        src_range[:, 0] = 0
        dst_range = torch.ones(batch_size, 2, self._input_size, device=device)
        dst_range[:, 0] = 0
        if self._batch_first:
            return (
                torch.rand(batch_size, seq_len, self._input_size, device=device),
                src_range,
                dst_range,
            )
        return (
            torch.rand(seq_len, batch_size, self._input_size, device=device),
            src_range,
            dst_range,
        )
