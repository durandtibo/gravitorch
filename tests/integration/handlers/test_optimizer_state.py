import os
from unittest.mock import Mock, patch

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer

from gravitorch import distributed as dist
from gravitorch.distributed import gloocontext
from gravitorch.handlers import ConsolidateOptimizerState
from tests.testing import gloo_available

###############################################
#     Tests for ConsolidateOptimizerState     #
###############################################


@gloo_available
@patch.dict(
    os.environ,
    {
        dist.MASTER_ADDR: "127.0.0.1",
        dist.MASTER_PORT: "29521",
        dist.WORLD_SIZE: "1",
        dist.RANK: "0",
        dist.LOCAL_RANK: "0",
    },
    clear=True,
)
def test_consolidate_optimizer_state_consolidate_zero():
    with gloocontext():
        optimizer = ZeroRedundancyOptimizer(
            params=torch.nn.Linear(4, 5).parameters(), optimizer_class=torch.optim.SGD, lr=0.01
        )
        ConsolidateOptimizerState().consolidate(engine=Mock(optimizer=optimizer))
        assert isinstance(optimizer.state_dict(), dict)
