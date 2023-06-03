__all__ = ["setup_clip_grad"]

import logging
from collections.abc import Callable
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def setup_clip_grad(config: Optional[dict]) -> tuple[Optional[Callable], tuple]:
    r"""Sets up the gradient clipping strategy.

    Args:
    ----
        config (dict): Specifies the configuration of the gradient
            clipping strategy.

    Returns:
    -------
        tuple: clip gradient function, clip gradient arguments.

    Raises:
    ------
        RuntimeError: if it is an invalid clipping gradient name.

    Example usage:

    .. code-block:: pycon

        >>> from gravitorch.loops.training.utils import setup_clip_grad
        >>> setup_clip_grad({"name": "clip_grad_value", "clip_value": 0.5})
        (<function torch.nn.utils.clip_grad.clip_grad_value_(parameters: Union[torch.Tensor, Iterable[torch.Tensor]], clip_value: float, foreach: Optional[bool] = None) -> None>,
         (0.5,))
        >>> setup_clip_grad({"name": "clip_grad_norm", "max_norm": 0.5, "norm_type": 2.0})
        (<function torch.nn.utils.clip_grad.clip_grad_norm_(parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor>,
         (0.5, 2.0))
        >>> setup_clip_grad({})
        (None, ())
    """
    config = config or {}
    if not config:
        return None, tuple()

    name = config["name"]
    if name == "clip_grad_value":
        clip_value = config.get("clip_value", 0.25)
        logger.info(f"clip gradient by value {clip_value}")
        return torch.nn.utils.clip_grad_value_, (clip_value,)
    if name == "clip_grad_norm":
        max_norm = config.get("max_norm", 1.0)
        norm_type = config.get("norm_type", 2.0)
        logger.info(f"clip gradient by maximum norm {max_norm} (norm type: {norm_type})")
        return torch.nn.utils.clip_grad_norm_, (max_norm, norm_type)
    raise RuntimeError(
        f"Incorrect clip grad name ({name}). The valid values are ``clip_grad_value`` "
        "and ``clip_grad_norm``"
    )
