__all__ = [
    "AsinhScalarEncoder",
    "AsinhCosSinScalarEncoder",
    "CosSinScalarEncoder",
    "ScalarEncoderFFN",
]

import math
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from gravitorch.nn.utils.factory import setup_module
from gravitorch.utils.tensor.misc import to_tensor


class AsinhScalarEncoder(Module):
    r"""Implements a scalar encoder using the inverse hyperbolic sine (arcsinh).

    Args:
        scale (``torch.nn.Tensor`` or list or tuple): Specifies
            the initial scale values.
        learnable (bool, optional): If ``True`` the scale parameters
            are learnable, otherwise they are frozen.
            Default: ``False``
    """

    def __init__(
        self,
        scale: Union[Tensor, list[float], tuple[float, ...]],
        learnable: bool = False,
    ):
        super().__init__()
        self.scale = Parameter(to_tensor(scale), requires_grad=learnable)

    @property
    def input_size(self) -> int:
        r"""int: The input feature size."""
        return 1

    def extra_repr(self) -> str:
        return f"dim={self.scale.shape[0]}, learnable={self.scale.requires_grad}"

    def forward(self, scalar: Tensor):
        r"""Computes a scalar representation.

        Args:
            scalar (``torch.Tensor`` of type float and shape
                ``(*, 1)``): Specifies the scalar values to encode.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(*, output_size)``: The scalar representation.
        """
        return scalar.mul(self.scale).asinh()

    @classmethod
    def create_rand_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> "AsinhScalarEncoder":
        r"""Creates a `AsinhScalarEncoder`` where the scales are uniformly
        initialized in the specified scale range.

        Args:
            dim (int): Specifies the dimension i.e. the number of
                scales.
            min_scale (float): Specifies the minimum scale.
            max_scale (float): Specifies the maximum scale.
            learnable (bool, optional): If ``True`` the scales are
                learnable, otherwise they are frozen.
                Default: ``False``

        Returns:
            ``AsinhScalarEncoder``: An instantiated
                ``AsinhScalarEncoder`` where the scales are
                uniformly initialized in a scale range.
        """
        if dim < 1:
            raise ValueError(f"dim has to be greater or equal to 1 (received: {dim})")
        if min_scale <= 0:
            raise ValueError(f"min_scale has to be greater than 0 (received: {min_scale})")
        if max_scale < min_scale:
            raise ValueError(
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
        return cls(
            scale=torch.rand(dim).mul(max_scale - min_scale).add(min_scale),
            learnable=learnable,
        )

    @classmethod
    def create_linspace_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> "AsinhScalarEncoder":
        r"""Creates a `AsinhScalarEncoder`` where the scales are evenly spaced.

        Args:
            dim (int): Specifies the dimension i.e. the number of
                scales.
            min_scale (float): Specifies the minimum scale.
            max_scale (float): Specifies the maximum scale.
            learnable (bool, optional): If ``True`` the scales are
                learnable, otherwise they are frozen.
                Default: ``False``

        Returns:
            ``AsinhScalarEncoder``: An instantiated
                ``AsinhScalarEncoder`` where the scales are
                evenly spaced.
        """
        if dim < 1:
            raise ValueError(f"dim has to be greater or equal to 1 (received: {dim})")
        if min_scale <= 0:
            raise ValueError(f"min_scale has to be greater than 0 (received: {min_scale})")
        if max_scale < min_scale:
            raise ValueError(
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
        return cls(
            scale=torch.linspace(start=min_scale, end=max_scale, steps=dim),
            learnable=learnable,
        )

    @classmethod
    def create_logspace_scale(
        cls,
        dim: int,
        min_scale: float,
        max_scale: float,
        learnable: bool = False,
    ) -> "AsinhScalarEncoder":
        r"""Creates a `AsinhScalarEncoder`` where the scales are evenly spaced
        in the log space.

        Args:
            dim (int): Specifies the dimension i.e. the number of
                scales.
            min_scale (float): Specifies the minimum scale.
            max_scale (float): Specifies the maximum scale.
            learnable (bool, optional): If ``True`` the scales are
                learnable, otherwise they are frozen.
                Default: ``False``

        Returns:
            ``AsinhScalarEncoder``: An instantiated
                ``AsinhScalarEncoder`` where the scales are
                evenly spaced in the log space.
        """
        if dim < 1:
            raise ValueError(f"dim has to be greater or equal to 1 (received: {dim})")
        if min_scale <= 0:
            raise ValueError(f"min_scale has to be greater than 0 (received: {min_scale})")
        if max_scale < min_scale:
            raise ValueError(
                f"max_scale has to be greater than min_scale {min_scale} "
                f"(received: {max_scale})"
            )
        return cls(
            scale=torch.logspace(start=math.log10(min_scale), end=math.log10(max_scale), steps=dim),
            learnable=learnable,
        )


class CosSinScalarEncoder(Module):
    r"""Implements a frequency/phase-shift scalar encoder where the periodic
    functions are cosine and sine.

    Args:
        frequency (``torch.nn.Tensor`` or list or tuple): Specifies
            the initial frequency values.
        phase_shift (``torch.nn.Tensor`` or list or tuple): Specifies
            the initial phase-shift values.
        learnable (bool, optional): If ``True`` the frequencies and
            phase-shift parameters are learnable, otherwise they are
            frozen. Default: ``False``
    """

    def __init__(
        self,
        frequency: Union[Tensor, list[float], tuple[float, ...]],
        phase_shift: Union[Tensor, list[float], tuple[float, ...]],
        learnable: bool = False,
    ):
        super().__init__()
        frequency = to_tensor(frequency)
        phase_shift = to_tensor(phase_shift)
        if frequency.ndim != 1:
            raise ValueError(
                f"Incorrect number of dimensions for `frequency` (shape={frequency.shape}). "
                f"`frequency` has to be a 1D tensor or equivalent."
            )
        if frequency.shape != phase_shift.shape:
            raise ValueError(
                f"Incorrect shapes. The shape of `frequency` (shape={frequency.shape})"
                f"does not match with `phase_shift` (shape={phase_shift.shape})"
            )
        self.frequency = Parameter(frequency, requires_grad=learnable)
        self.phase_shift = Parameter(phase_shift, requires_grad=learnable)
        self._half_size = int(self.frequency.shape[0] // 2)

    @property
    def input_size(self) -> int:
        r"""int: The input feature size."""
        return 1

    def extra_repr(self) -> str:
        return f"dim={self.frequency.shape[0]}, learnable={self.frequency.requires_grad}"

    def forward(self, scalar: Tensor):
        r"""Computes a scalar representation.

        Args:
            scalar (``torch.Tensor`` of type float and shape
                ``(*, 1)``): Specifies the scalar values to encode.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(*, output_size)``: The scalar representation.
        """
        features = self.frequency * scalar + self.phase_shift
        return torch.cat(
            (features[..., : self._half_size].sin(), features[..., self._half_size :].cos()),
            dim=-1,
        )

    @classmethod
    def create_rand_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are uniformly
        initialized in a frequency range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_frequency (float): Specifies the minimum frequency.
            max_frequency (float): Specifies the maximum frequency.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                uniformly initialized in a frequency range.
        """
        if num_frequencies < 1:
            raise ValueError(
                f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            )
        if min_frequency <= 0:
            raise ValueError(f"min_frequency has to be greater than 0 (received: {min_frequency})")
        if max_frequency < min_frequency:
            raise ValueError(
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
        return cls(
            frequency=torch.rand(num_frequencies)
            .mul(max_frequency - min_frequency)
            .add(min_frequency)
            .repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_rand_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are uniformly
        initialized for a given value range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_abs_value (float): Specifies the minimum absolute
                value to encode.
            max_abs_value (float): Specifies the maximum absolute
                value to encoder.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                uniformly initialized for a given value range.
        """
        return cls.create_rand_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )

    @classmethod
    def create_linspace_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are evenly
        spaced in a frequency range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_frequency (float): Specifies the minimum frequency.
            max_frequency (float): Specifies the maximum frequency.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                evenly spaced in a frequency range.
        """
        if num_frequencies < 1:
            raise ValueError(
                f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            )
        if min_frequency <= 0:
            raise ValueError(f"min_frequency has to be greater than 0 (received: {min_frequency})")
        if max_frequency < min_frequency:
            raise ValueError(
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
        return cls(
            frequency=torch.linspace(
                start=min_frequency, end=max_frequency, steps=num_frequencies
            ).repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_linspace_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are evenly
        spaced given a value range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_abs_value (float): Specifies the minimum absolute
                value to encode.
            max_abs_value (float): Specifies the maximum absolute
                value to encoder.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                evenly spaced.
        """
        return cls.create_linspace_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )

    @classmethod
    def create_logspace_frequency(
        cls,
        num_frequencies: int,
        min_frequency: float,
        max_frequency: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are evenly
        spaced in the log space in a frequency range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_frequency (float): Specifies the minimum frequency.
            max_frequency (float): Specifies the maximum frequency.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                evenly spaced in the log space in a frequency range.
        """
        if num_frequencies < 1:
            raise ValueError(
                f"num_frequencies has to be greater or equal to 1 (received: {num_frequencies})"
            )
        if min_frequency <= 0:
            raise ValueError(f"min_frequency has to be greater than 0 (received: {min_frequency})")
        if max_frequency < min_frequency:
            raise ValueError(
                f"max_frequency has to be greater than min_frequency {min_frequency} "
                f"(received: {max_frequency})"
            )
        return cls(
            frequency=torch.logspace(
                start=math.log10(min_frequency),
                end=math.log10(max_frequency),
                steps=num_frequencies,
            ).repeat(2),
            phase_shift=torch.zeros(2 * num_frequencies),
            learnable=learnable,
        )

    @classmethod
    def create_logspace_value_range(
        cls,
        num_frequencies: int,
        min_abs_value: float,
        max_abs_value: float,
        learnable: bool = True,
    ) -> "CosSinScalarEncoder":
        r"""Creates a `CosSinScalarEncoder`` where the frequencies are evenly
        spaced in the log space given a value range.

        Args:
            num_frequencies (int): Specifies the number of
                frequencies.
            min_abs_value (float): Specifies the minimum absolute
                value to encode.
            max_abs_value (float): Specifies the maximum absolute
                value to encoder.
            learnable (bool, optional): If ``True`` the parameters
                are learnable, otherwise they are frozen.
                Default: ``True``

        Returns:
            ``CosSinScalarEncoder``: An instantiated
                ``CosSinScalarEncoder`` where the frequencies are
                evenly spaced in the log space.
        """
        return cls.create_logspace_frequency(
            num_frequencies=num_frequencies,
            min_frequency=1 / max_abs_value,
            max_frequency=1 / min_abs_value,
            learnable=learnable,
        )


class AsinhCosSinScalarEncoder(CosSinScalarEncoder):
    r"""Extension of ``CosSinScalarEncoder`` with an extra feature:

    the inverse hyperbolic sine (arcsinh) of the input.
    """

    def forward(self, scalar: Tensor):
        r"""Computes a scalar representation.

        Args:
            scalar (``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn, 1)``): Specifies the scalar
                values to encode.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(d0, d1, ..., dn, output_size)``: The scalar
                representation.
        """
        features = self.frequency * scalar + self.phase_shift
        return torch.cat(
            (
                features[..., : self._half_size].sin(),
                features[..., self._half_size :].cos(),
                scalar.asinh(),
            ),
            dim=-1,
        )


class ScalarEncoderFFN(Module):
    r"""Implements a scalar encoder with a feed-forward network.

    Args:
        encoder (``torch.nn.Module`` or dict): Specifies the encoder
            or its configuration.
        ffn (``torch.nn.Module`` or dict): Specifies the feed-forward
            network or its configuration.
    """

    def __init__(self, encoder: Union[Module, dict], ffn: Union[Module, dict]):
        super().__init__()
        self.encoder = setup_module(encoder)
        self.ffn = setup_module(ffn)

    @property
    def input_size(self) -> int:
        r"""int: The input feature size."""
        return self.encoder.input_size

    def forward(self, scalar: Tensor) -> torch.Tensor:
        r"""Computes a scalar representation.

        Args:
            scalar (``torch.Tensor`` of type float and shape
                ``(*, 1)``): Specifies the scalar values to encode.

        Returns:
            ``torch.Tensor`` of type float and shape
                ``(*, output_size)``: The scalar representation.
        """
        return self.ffn(self.encoder(scalar))
