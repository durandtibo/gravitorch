__all__ = ["app_vae_time_loss"]

import torch

from gravitorch.nn.functional.loss_helpers import basic_loss_reduction


def app_vae_time_loss(
    lmbda: torch.Tensor,
    target: torch.Tensor,
    log_input: bool = False,
    delta: float = 0.1,
    eps: float = 1e-8,
    max_log_value: float = 20.0,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Computes the APP-VAE time loss value.

    Args:
        lmbda (``torch.Tensor`` of type float): Specifies the
            predicted lambda i.e. the rate of the Exponential
            distribution.
        target (``torch.Tensor`` of type float and same shape as
            ``lmbda``): Specifies the target values.
        log_input (bool, optional):  If ``True``, the expected input
            is ``log(lambda)``, otherwise it is ``lambda``.
            Default: ``False``
        delta (float, optional): Specifies a small time interval used
            to compute the NLL. Default: ``0.1``
        eps (float, optional): Small value to avoid evaluation of
            :math:`\log(0)`. Default: ``1e-8``
        max_log_value (float, optional): Specifies the maximum value
            used to clip ``log_rate`` before to compute the
            exponential when :attr:`log_input = True`.
            Default: ``20.0``
        reduction (string, optional): Specifies the reduction to apply
            to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Returns:
        ``torch.Tensor`` of type float: The negative log-likelihood
            with Exponential distribution of target. The shape of
            the tensor depends on the reduction strategy.
    """
    if delta <= 0:
        raise ValueError(f"delta has to be greater than 0 but received: {delta}")
    if log_input:
        lmbda = lmbda.clamp(max=max_log_value).exp()
    nll = lmbda.mul(target) - (1 - lmbda.mul(-delta).exp() + eps).log()
    return basic_loss_reduction(nll, reduction)
