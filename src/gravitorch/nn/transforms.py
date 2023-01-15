__all__ = [
    "Asinh",
    "Isymlog",
    "Log1p",
    "OnePolynomial",
    "Safeexp",
    "Safelog",
    "SequenceToBatch",
    "Sinh",
    "Squeeze",
    "Symlog",
    "ToBinaryLabel",
    "ToCategoricalLabel",
    "ToFloat",
    "ToLong",
]

import math
from typing import Union

from torch import Tensor
from torch.nn import Module

from gravitorch.utils.tensor.math_ops import isymlog, safeexp, safelog, symlog


class Asinh(Module):
    r"""Implements a ``torch.nn.Module`` to compute the inverse hyperbolic sine
    (arcsinh) of the elements."""

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.asinh()


class Sinh(Module):
    r"""Implements a ``torch.nn.Module`` to compute the hyperbolic sine (sinh)
    of the elements."""

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.sinh()


class Log1p(Module):
    r"""Implements a ``torch.nn.Module`` to compute the natural logarithm of
    ``(1 + input)``."""

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.log1p()


class OnePolynomial(Module):
    r"""Implements a ``torch.nn.Module`` to compute a polynomial transformation
    with a single term.

    The equation is ``alpha * x^gamma + beta`` where ``x`` is the
    module input tensor.

    Args:
        alpha (float, optional): Specifies the alpha value.
            Default: ``1.0``
        beta (float, optional): Specifies the beta value.
            Default: ``0.0``
        gamma (float, optional): Specifies the gamma value.
            Default: ``1.0``
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)

    def extra_repr(self) -> str:
        return f"alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}"

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Computes a polynomial transformation of the input tensor.

        Args:
            tensor (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)``): Specifies the tensor of values
                to transform.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)``: The transformed values.
        """
        return tensor.pow(self._gamma).mul(self._alpha).add(self._beta)

    @classmethod
    def create_from_range(
        cls,
        gamma: float = 1.0,
        input_min_value: float = 0.0,
        input_max_value: float = 1.0,
        output_min_value: float = 0.0,
        output_max_value: float = 1.0,
    ) -> "OnePolynomial":
        r"""Instantiates a polynomial transform module for a given input and
        output ranges.

        Args:
            gamma (float, optional): Specifies the gamma value.
                Default: ``1.0``
            input_min_value (float, optional): Specifies the expected
                minimum input value. Default: ``0.0``
            input_max_value (float, optional): Specifies the expected
                maximum input value. Default: ``1.0``
            output_min_value (float, optional): Specifies the expected
                minimum output value. Default: ``0.0``
            output_max_value (float, optional): Specifies the expected
                maximum output value. Default: ``1.0``

        Returns:
            ``OnePolynomial``: An instantiated polynomial transform
                module.
        """
        alpha = (output_max_value - output_min_value) / (
            math.pow(input_max_value, gamma) - math.pow(input_min_value, gamma)
        )
        return cls(
            alpha=alpha,
            beta=output_min_value - alpha * math.pow(input_min_value, gamma),
            gamma=gamma,
        )


class Safeexp(Module):
    r"""Implements a ``torch.nn.Module`` to compute the exponential of the
    elements.

    The values that are higher than the specified minimum value are
    set to this maximum value. Using a not too large positive value
    leads to an output tensor without Inf.

    Args:
        max_value (float, optional): Specifies the maximum value.
            Default: ``20.0``
    """

    def __init__(self, max_value: float = 20.0):
        super().__init__()
        self._max_value = float(max_value)

    @property
    def max_value(self) -> float:
        r"""float: The maximum value before to compute the exponential."""
        return self._max_value

    def extra_repr(self) -> str:
        return f"max_value={self._max_value}"

    def forward(self, tensor: Tensor) -> Tensor:
        return safeexp(tensor, self._max_value)


class Safelog(Module):
    r"""Implements a ``torch.nn.Module`` to compute the logarithm natural of the
    elements.

    The values that are lower than the specified minimum value are set
    to this minimum value. Using a small positive value leads to an
    output tensor without NaN or Inf.

    Args:
        min_value (float, optional): Specifies the minimum value.
            Default: ``1e-8``
    """

    def __init__(self, min_value: float = 1e-8):
        super().__init__()
        self._min_value = float(min_value)

    @property
    def min_value(self) -> float:
        r"""float: The minimum value before to compute the exponential."""
        return self._min_value

    def extra_repr(self) -> str:
        return f"min_value={self._min_value}"

    def forward(self, tensor: Tensor) -> Tensor:
        return safelog(tensor, self._min_value)


class Squeeze(Module):
    r"""Implements a ``torch.nn.Module`` to squeeze the input tensor.

    Args:
        dim (int or ``None``, optional): Specifies the dimension to
            squeeze the input tensor. If ``None``, all the dimensions
            of the input tensor of size 1 are removed.
            Default: ``None``
    """

    def __init__(self, dim: Union[int, None] = None):
        super().__init__()
        self._dim = dim

    def extra_repr(self) -> str:
        return f"dim={self._dim}"

    def forward(self, tensor: Tensor) -> Tensor:
        if self._dim is None:
            return tensor.squeeze()
        return tensor.squeeze(self._dim)


class Symlog(Module):
    r"""Implements a ``torch.nn.Module`` to compute the symmetric logarithm
    natural of the elements."""

    def forward(self, tensor: Tensor) -> Tensor:
        return symlog(tensor)


class Isymlog(Module):
    r"""Implements a ``torch.nn.Module`` to compute the inverse symmetric
    logarithm natural of the elements."""

    def forward(self, tensor: Tensor) -> Tensor:
        return isymlog(tensor)


class ToBinaryLabel(Module):
    r"""Implements a ``torch.nn.Module`` to compute binary labels from scores by
    thresholding.

    Args:
        threshold (float, optional): Specifies the threshold value
            used to compute the binary labels.
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        r"""float: The threshold used to compute the binary label."""
        return self._threshold

    def extra_repr(self) -> str:
        return f"threshold={self._threshold}"

    def forward(self, scores: Tensor) -> Tensor:
        r"""Computes binary labels from scores.

        Args:
            scores (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)``): Specifies the scores used to
                compute the binary labels.

        Returns:
            ``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``. The computed binary labels.
                The values are ``0`` and ``1``.
        """
        return (scores > self._threshold).long()


class ToFloat(Module):
    r"""Implements a ``torch.nn.Module`` to convert a tensor to a float
    tensor."""

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Converts a tensor to a float tensor.

        Args:
            tensor (``torch.Tensor`` of shape ``(d0, d1, ..., dn)``):
                Specifies the input tensor.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn)``. The converted float tensor.
        """
        return tensor.float()


class ToLong(Module):
    r"""Implements a ``torch.nn.Module`` to convert a tensor to a long
    tensor."""

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Converts a tensor to a long tensor.

        Args:
            tensor (``torch.Tensor`` of shape ``(d0, d1, ..., dn)``):
                Specifies the input tensor.

        Returns:
            ``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``. The converted long tensor.
        """
        return tensor.long()


class ToCategoricalLabel(Module):
    r"""Implements a ``torch.nn.Module`` to compute categorical labels from
    scores."""

    def forward(self, scores: Tensor) -> Tensor:
        r"""Computes categorical labels from scores.

        Args:
            scores (``torch.Tensor`` of shape
                ``(d0, d1, ..., dn, num_classes)`` and type float):
                Specifies the scores used to compute the categorical
                labels.

        Returns:
            ``torch.Tensor`` of type long and shape
                ``(d0, d1, ..., dn)``. The computed categorical labels.
                The values are in ``{0, 1, ..., num_classes-1}``.
        """
        return scores.argmax(dim=-1, keepdim=False)


class SequenceToBatch(Module):
    r"""Implements a ``torch.nn.Module`` to convert a batch of sequences to a
    batch i.e. remove the sequence dimension.

    This module supports batch first and sequence first format.
    """

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Convert a batch of sequences to a batch.

        Args:
            tensor (``torch.Tensor`` of shape
                ``(batch_size, seq_len, d2, ..., dn)`` or
                ``(seq_len, batch_size, d2, ..., dn)``):
                Specifies the tensor to convert.

        Returns:
            ``torch.Tensor`` of shape
                ``(batch_size * seq_len, d2, ..., dn)``:
                The converted tensor.
        """
        return tensor.view(-1, *tensor.shape[2:])
