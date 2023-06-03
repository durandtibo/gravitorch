from typing import Optional

import torch
from pytest import mark, raises

from gravitorch.loops.training.utils import setup_clip_grad

#####################################
#     Tests for setup_clip_grad     #
#####################################


@mark.parametrize("config", (None, {}))
def test_setup_clip_grad_empty(config: Optional[dict]) -> None:
    assert setup_clip_grad(config) == (None, tuple())


@mark.parametrize("clip_value", (0.1, 0.25, 1.0))
def test_setup_clip_grad_clip_grad_value(clip_value: float) -> None:
    assert setup_clip_grad({"name": "clip_grad_value", "clip_value": clip_value}) == (
        torch.nn.utils.clip_grad_value_,
        (clip_value,),
    )


def test_setup_clip_grad_clip_grad_value_default() -> None:
    assert setup_clip_grad({"name": "clip_grad_value"}) == (
        torch.nn.utils.clip_grad_value_,
        (0.25,),
    )


@mark.parametrize("max_norm", (0.1, 0.25, 1.0))
@mark.parametrize("norm_type", (1.0, 2.0, 3.0))
def test_setup_clip_grad_clip_grad_norm(max_norm: float, norm_type: float) -> None:
    assert setup_clip_grad(
        {"name": "clip_grad_norm", "max_norm": max_norm, "norm_type": norm_type}
    ) == (
        torch.nn.utils.clip_grad_norm_,
        (max_norm, norm_type),
    )


def test_setup_clip_grad_clip_grad_norm_default() -> None:
    assert setup_clip_grad({"name": "clip_grad_norm"}) == (
        torch.nn.utils.clip_grad_norm_,
        (1.0, 2.0),
    )


def test_setup_clip_grad_incorrect_name() -> None:
    with raises(RuntimeError, match="Incorrect clip grad name"):
        setup_clip_grad({"name": "incorrect"})
