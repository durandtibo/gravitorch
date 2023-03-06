import torch
from coola import objects_are_allclose
from pytest import fixture
from torch import nn
from torch.optim import Optimizer

from gravitorch.lr_schedulers import InverseSquareRootLR

BASE_LR = 0.01


@fixture
def optimizer() -> Optimizer:
    return torch.optim.SGD(nn.Linear(4, 6).parameters(), lr=BASE_LR)


########################################
#     Tests for InverseSquareRootLR    #
########################################


def test_inverse_square_root_lr(optimizer: Optimizer) -> None:
    scheduler = InverseSquareRootLR(optimizer)
    lrs = []
    for _ in range(11):
        lrs.append(scheduler.get_last_lr())
        optimizer.step()
        scheduler.step()
    assert objects_are_allclose(lrs[0], [0.01])
    assert objects_are_allclose(lrs[4], [0.004472135954999579])
    assert objects_are_allclose(lrs[9], [0.0031622776601683794])
