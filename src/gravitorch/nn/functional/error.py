__all__ = ["absolute_error", "absolute_relative_error", "symmetric_absolute_relative_error"]

from torch import Tensor


def absolute_error(prediction: Tensor, target: Tensor) -> Tensor:
    r"""Computes the absolute error between the predictions and targets.

    Args:
        prediction (``torch.Tensor``): Specifies the predictions.
        target (``torch.Tensor`` with same shape and data type as
            ``prediction``): Specifies the target tensor.

    Returns:
        ``torch.Tensor`` with same shape and data type as
            the inputs: The absolute error.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import absolute_error
        >>> absolute_error(torch.eye(2), torch.ones(2, 2))
        tensor([[0., 1.],
                [1., 0.]])
    """
    return target.sub(prediction).abs()


def absolute_relative_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Computes the absolute relative error between the predictions and
    targets.

    Args:
        prediction (``torch.Tensor``): Specifies the predictions.
        target (``torch.Tensor`` with same shape and data type as
            ``prediction``): Specifies the target tensor.
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Returns:
        ``torch.Tensor`` with same shape and data type as
            the inputs: The absolute relative error.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import absolute_relative_error
        >>> absolute_relative_error(torch.eye(2), torch.ones(2, 2))
        tensor([[0., 1.],
                [1., 0.]])
    """
    return target.sub(prediction).div(target.abs().clamp(min=eps)).abs()


def symmetric_absolute_relative_error(
    prediction: Tensor, target: Tensor, eps: float = 1e-8
) -> Tensor:
    r"""Computes the symmetric absolute relative error between the predictions
    and targets.

    Args:
        prediction (``torch.Tensor``): Specifies the predictions.
        target (``torch.Tensor`` with same shape and data type as
            ``prediction``): Specifies the target tensor.
        eps (float, optional): Specifies an arbitrary small strictly
            positive number to avoid undefined results when the target
            is zero. Default: ``1e-8``

    Returns:
        ``torch.Tensor`` with same shape and data type as
            the inputs: The symmetric absolute relative error.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import symmetric_absolute_relative_error
        >>> symmetric_absolute_relative_error(torch.eye(2), torch.ones(2, 2))
        tensor([[0., 1.],
                [1., 0.]])
    """
    return (
        target.sub(prediction).div(target.abs().add(prediction.abs()).mul(0.5).clamp(min=eps)).abs()
    )
