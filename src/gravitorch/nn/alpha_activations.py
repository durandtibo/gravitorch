__all__ = [
    "BaseAlphaActivation",
    "ExpSin",
    "Gaussian",
    "Laplacian",
    "MultiQuadratic",
    "Quadratic",
    "Sin",
]

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class BaseAlphaActivation(Module):
    r"""Defines a base class to implement an activation layer with a learnable
    parameter ``alpha``.

    When called without arguments, the activation layer uses a single
    parameter ``alpha`` across all input channels. If called with a
    first argument, a separate ``alpha`` is used for each input
    channel.

    Args:
        num_parameters (int, optional): Specifies the number of
            learnable parameters. Although it takes an integer as
            input, there is only two values are legitimate: ``1``,
            or the number of channels at input. Default: ``1``
        init (float, optional): Specifies the initial value of the
            learnable parameter(s). Default: ``1.0``
        learnable (bool, optional): If ``True``, the parameters are
            learnt during the training, otherwise they are fixed.
            Default: ``True``
    """

    def __init__(self, num_parameters: int = 1, init: float = 1.0, learnable: bool = True):
        super().__init__()
        self.alpha = Parameter(
            torch.full((num_parameters,), init, dtype=torch.float), requires_grad=learnable
        )

    def extra_repr(self) -> str:
        return f"num_parameters={self.alpha.numel()}, learnable={self.alpha.requires_grad}"


class Gaussian(BaseAlphaActivation):
    r"""Implements the Gaussian activation layer.

    Formula: ``exp(-0.5 * x^2 / alpha^2)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.pow(2).mul(-0.5).div(self.alpha.pow(2)).exp()


class Laplacian(BaseAlphaActivation):
    r"""Implements the Laplacian activation layer.

    Formula: ``exp(-|x| / alpha)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.abs().mul(-1).div(self.alpha).exp()


class ExpSin(BaseAlphaActivation):
    r"""Implements the ExpSin activation layer.

    Formula: ``exp(-sin(alpha * x))``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.mul(self.alpha).sin().exp()


class Sin(BaseAlphaActivation):
    r"""Implements the sine activation layer.

    Formula: ``sin(alpha * x)``
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor.mul(self.alpha).sin()


class Quadratic(BaseAlphaActivation):
    r"""Implements the Quadratic activation layer.

    Formula: ``1 / (1 + (alpha * x)^2)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return 1.0 / tensor.mul(self.alpha).pow(2).add(1)


class MultiQuadratic(BaseAlphaActivation):
    r"""Implements the Multi Quadratic activation layer.

    Formula: ``1 / sqrt(1 + (alpha * x)^2)``

    This activation layer was proposed in the following paper:

        Beyond Periodicity: Towards a Unifying Framework for Activations in Coordinate-MLPs.
        Ramasinghe S., Lucey S.
        ECCV 2022. (http://arxiv.org/pdf/2111.15135)
    """

    def forward(self, tensor: Tensor) -> Tensor:
        return 1.0 / tensor.mul(self.alpha).pow(2).add(1).sqrt()
