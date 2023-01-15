__all__ = ["AppVAETimeLoss"]

import torch
from torch.nn import Module

from gravitorch.nn.functional import check_basic_loss_reduction
from gravitorch.nn.functional.experimental import app_vae_time_loss


class AppVAETimeLoss(Module):
    r"""Implements a loss function to compute the APP-VAE time loss.

    Args:
        log_input (bool, optional): If ``True``, the expected input is
            ``log(lambda)``, otherwise it is ``lambda``.
            Default: ``False``
        delta (float, optional): Specifies a small time interval used
            to compute the NLL. Default: ``0.1``
        eps (float, optional): Small value to avoid evaluation of
            :math:`\log(0)` when :attr:`log_input = False`.
            Default: ``1e-8``
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
    """

    def __init__(
        self,
        log_input: bool = False,
        delta: float = 0.1,
        eps: float = 1e-8,
        max_log_value: float = 20.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self._log_input = bool(log_input)

        if delta < 0:
            raise ValueError(f"delta has to be greater or equal to 0 but received {delta}")
        self._delta = float(delta)
        if eps < 0:
            raise ValueError(f"eps has to be greater or equal to 0 but received {eps}")
        self._eps = float(eps)
        self._max_log_value = float(max_log_value)

        check_basic_loss_reduction(reduction)
        self.reduction = str(reduction)

    def extra_repr(self) -> str:
        return (
            f"log_input={self._log_input}, delta={self._delta}, eps={self._eps}, "
            f"max_log_value={self._max_log_value}, reduction={self.reduction}"
        )

    def forward(self, lmbda: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computes the APP-VAE time loss value.

        Args:
            lmbda (``torch.Tensor`` of type float): Specifies the
                predicted lambda i.e. the rate of the Exponential
                distribution.
            target (``torch.Tensor`` of type float and same shape as
                ``lmbda``): Specifies the target values.

        Returns:
            ``torch.Tensor`` of type float: The negative
                log-likelihood with Exponential distribution of
                target. The shape of the tensor depends on the
                reduction strategy.
        """
        return app_vae_time_loss(
            lmbda=lmbda,
            target=target,
            log_input=self._log_input,
            delta=self._delta,
            eps=self._eps,
            max_log_value=self._max_log_value,
            reduction=self.reduction,
        )
