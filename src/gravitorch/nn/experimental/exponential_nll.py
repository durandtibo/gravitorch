__all__ = ["ExponentialNLLLoss"]

from torch import Tensor
from torch.nn import Module

from gravitorch.nn.functional.experimental import exponential_nll_loss
from gravitorch.nn.functional.loss_helpers import check_basic_loss_reduction


class ExponentialNLLLoss(Module):
    r"""Implements a criterion to compute the negative log-likelihood (NLL) with
    Exponential distribution of target.

    Args:
        log_input (bool, optional): If ``True``, the expected input
            is ``log(lambda)``, otherwise it is ``lambda``.
            Default: ``True``
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
        log_input: bool = True,
        eps: float = 1e-8,
        max_log_value: float = 20.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self._log_input = bool(log_input)

        if eps < 0:
            raise ValueError(f"eps has to be greater or equal to 0 but received {eps}")
        self._eps = float(eps)
        self._max_log_value = float(max_log_value)

        check_basic_loss_reduction(reduction)
        self.reduction = str(reduction)

    def extra_repr(self) -> str:
        return (
            f"log_input={self._log_input}, eps={self._eps}, "
            f"max_log_value={self._max_log_value}, reduction={self.reduction}"
        )

    def forward(self, log_rate: Tensor, target: Tensor) -> Tensor:
        r"""Computes the negative log-likelihood (NLL) with Exponential
        distribution of target.

        Args:
            log_rate (``torch.Tensor`` of type float): Specifies the
                predicted rates (``lambda``).
            target (``torch.Tensor`` of type float and same shape as
                ``log_rate``): Specifies the target values.

        Returns:
            ``torch.Tensor`` of type float: The negative
                log-likelihood with Exponential distribution of
                target. The shape of the tensor depends on the
                reduction strategy.
        """
        return exponential_nll_loss(
            log_rate=log_rate,
            target=target,
            log_input=self._log_input,
            eps=self._eps,
            max_log_value=self._max_log_value,
            reduction=self.reduction,
        )
