import torch
from coola import objects_are_allclose
from objectory import OBJECT_TARGET
from pytest import fixture, mark
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from gravitorch.lr_schedulers import (
    create_linear_warmup_cosine_decay_lr,
    create_linear_warmup_linear_decay_lr,
    create_sequential_lr,
)

BASE_LR = 0.01


@fixture
def optimizer() -> Optimizer:
    return torch.optim.SGD(nn.Linear(4, 6).parameters(), lr=BASE_LR)


#########################################
#     Tests for create_sequential_lr    #
#########################################


def test_create_sequential_lr(optimizer: Optimizer) -> None:
    scheduler = create_sequential_lr(
        optimizer,
        schedulers=[
            {
                OBJECT_TARGET: "torch.optim.lr_scheduler.LinearLR",
                "start_factor": 0.001,
                "end_factor": 1.0,
                "total_iters": 2,
            },
            LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=8),
        ],
        milestones=[2],
    )
    assert isinstance(scheduler, SequentialLR)
    lrs = []
    for _ in range(12):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [1e-5])
    assert objects_are_allclose(lrs[2], [BASE_LR])
    assert objects_are_allclose(lrs[10:12], [[1e-5], [1e-5]])


#########################################################
#     Tests for create_linear_warmup_cosine_decay_lr    #
#########################################################


@mark.parametrize("num_warmup_steps", (2, 5))
def test_create_linear_warmup_cosine_decay_lr_num_warmup_steps(
    optimizer: Optimizer, num_warmup_steps: int
) -> None:
    scheduler = create_linear_warmup_cosine_decay_lr(
        optimizer, num_warmup_steps=num_warmup_steps, num_total_steps=10
    )
    assert isinstance(scheduler, SequentialLR)
    lrs = []
    for _ in range(12):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [1e-5])
    assert objects_are_allclose(lrs[num_warmup_steps], [BASE_LR])
    assert objects_are_allclose(lrs[10], [1e-6])


@mark.parametrize("start_factor", (0.1, 0.01))
@mark.parametrize("end_lr", (0.1, 0.01))
def test_create_linear_warmup_cosine_decay_lr_factors(
    optimizer: Optimizer, start_factor: float, end_lr: float
) -> None:
    scheduler = create_linear_warmup_cosine_decay_lr(
        optimizer, num_warmup_steps=2, num_total_steps=10, start_factor=start_factor, end_lr=end_lr
    )
    assert isinstance(scheduler, SequentialLR)
    lrs = []
    for _ in range(12):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [BASE_LR * start_factor])
    assert objects_are_allclose(lrs[2], [BASE_LR])
    assert objects_are_allclose(lrs[10], [end_lr])


#########################################################
#     Tests for create_linear_warmup_linear_decay_lr    #
#########################################################


@mark.parametrize("num_warmup_steps", (2, 5))
def test_create_linear_warmup_linear_decay_lr_num_warmup_steps(
    optimizer: Optimizer, num_warmup_steps: int
) -> None:
    scheduler = create_linear_warmup_linear_decay_lr(
        optimizer, num_warmup_steps=num_warmup_steps, num_total_steps=10
    )
    assert isinstance(scheduler, SequentialLR)
    lrs = []
    for _ in range(12):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [1e-5])
    assert objects_are_allclose(lrs[num_warmup_steps], [BASE_LR])
    assert objects_are_allclose(lrs[10:12], [[1e-5], [1e-5]])


@mark.parametrize("start_factor", (0.1, 0.01))
@mark.parametrize("end_factor", (0.1, 0.01))
def test_create_linear_warmup_linear_decay_lr_factors(
    optimizer: Optimizer, start_factor: float, end_factor: float
) -> None:
    scheduler = create_linear_warmup_linear_decay_lr(
        optimizer,
        num_warmup_steps=2,
        num_total_steps=10,
        start_factor=start_factor,
        end_factor=end_factor,
    )
    assert isinstance(scheduler, SequentialLR)
    lrs = []
    for _ in range(12):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [BASE_LR * start_factor])
    assert objects_are_allclose(lrs[2], [BASE_LR])
    assert objects_are_allclose(lrs[10:12], [[BASE_LR * end_factor], [BASE_LR * end_factor]])
