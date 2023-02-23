__all__ = ["basic_loss_reduction", "check_basic_loss_reduction"]

from torch import Tensor

VALID_REDUCTIONS = ("none", "mean", "sum")


def basic_loss_reduction(tensor: Tensor, reduction: str) -> Tensor:
    r"""Computes a basic reduction for an input tensor.

    This function is designed to be used with loss functions.

    Args:
        tensor (``torch.Tensor``): Specifies the input tensor to
            reduce.
        reduction (str): Specifies the reduction strategy. The valid
            values are ``'mean'``, ``'none'``,  and ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum will be divided by the number of elements in the
            input, ``'sum'``: the output will be summed.

    Returns:
        ``torch.Tensor``: The reduced tensor. The shape of the tensor
            depends on the reduction strategy.

    Raises:
        ValueError if the reduction is not valid.

    Example usage:

    .. code-block:: python

        >>> import torch
        >>> from gravitorch.nn.functional import basic_loss_reduction
        >>> tensor = torch.arange(6).view(2, 3)
        >>> basic_loss_reduction(tensor, 'none')
        tensor([[0, 1, 2],
                [3, 4, 5]])
        >>> basic_loss_reduction(tensor, 'sum')
        tensor(15)
        >>> basic_loss_reduction(tensor, 'mean')
        tensor(2.5000)
    """
    if reduction == "mean":
        return tensor.float().mean()
    if reduction == "sum":
        return tensor.sum()
    if reduction == "none":
        return tensor
    raise ValueError(
        f"Incorrect reduction: {reduction}. The possible reductions are {VALID_REDUCTIONS}."
    )


def check_basic_loss_reduction(reduction: str) -> None:
    r"""Checks if the provided reduction ia a valid basic loss reduction.

    The valid reduction values are ``'mean'``, ``'none'``,  and
    ``'sum'``.

    Args:
        reduction (str): Specifies the reduction to check.

    Raises:
        ValueError if the provided reduction is not valid.

    Example usage:

    .. code-block:: python

        >>> from gravitorch.nn.functional import check_basic_loss_reduction
        >>> check_basic_loss_reduction('mean')
        >>> check_basic_loss_reduction('incorrect')
        Traceback (most recent call last): ...
    """
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Incorrect reduction: {reduction}. The valid reductions are: {VALID_REDUCTIONS}"
        )
