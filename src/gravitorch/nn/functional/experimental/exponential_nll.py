__all__ = ["exponential_nll_loss"]

from torch import Tensor

from gravitorch.nn.functional.loss_helpers import basic_loss_reduction


def exponential_nll_loss(
    log_rate: Tensor,
    target: Tensor,
    log_input: bool = True,
    eps: float = 1e-8,
    max_log_value: float = 20.0,
    reduction: str = "mean",
) -> Tensor:
    r"""Computes the negative log-likelihood (NLL) with Exponential distribution
    of target.

    Args:
        log_rate (``torch.Tensor`` of type float): Specifies the
            predicted rates (``lambda``).
        target (``torch.Tensor`` of type float and same shape as
            ``log_rate``): Specifies the target values.
        log_input (bool, optional): If ``True``, the expected input
            is ``log(rate)``, otherwise it is ``rate``.
            Default: ``True``
        eps (float, optional): Small value to avoid evaluation of
            :math:`\log(0)` when :attr:`log_input = False`.
            Default: ``1e-8``
        max_log_value (float, optional): Specifies the maximum value
            used to clip ``log_rate`` before to compute the
            exponential when :attr:`log_input = True`.
            Default: ``20.0``
        reduction (string, optional): Specifies the reduction to
            apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the
            sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be
            summed. Default: ``'mean'``

    Returns:
        ``torch.Tensor`` of type float: The negative log-likelihood
            with Exponential distribution of target. The shape of the
            tensor depends on the reduction strategy.
    """
    if log_input:
        # Input is log(rate)
        nll = log_rate.clamp(max=max_log_value).exp().mul(target) - log_rate
    else:
        # Input is rate
        nll = log_rate.mul(target) - log_rate.add(eps).log()
    return basic_loss_reduction(nll, reduction)
