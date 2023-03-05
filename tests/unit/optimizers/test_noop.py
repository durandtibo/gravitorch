from torch import nn

from gravitorch.optimizers.noop import NoOpOptimizer

###################################
#     Tests for NoOpOptimizer     #
###################################


def test_noop_optimizer_load_state_dict() -> None:
    optim = NoOpOptimizer(nn.Linear(4, 6).parameters())
    optim.load_state_dict({})


def test_noop_optimizer_state_dict() -> None:
    assert NoOpOptimizer(nn.Linear(4, 6).parameters()).state_dict() == {}


def test_noop_optimizer_step() -> None:
    net = nn.Linear(4, 6)
    weight = net.weight
    optim = NoOpOptimizer(net.parameters())
    optim.step()
    assert net.weight.equal(weight)
