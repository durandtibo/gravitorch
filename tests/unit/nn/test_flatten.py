from unittest.mock import Mock

import torch
from coola import objects_are_equal
from torch import nn

from gravitorch.nn import FlattenModule, MulticlassFlattenModule

###################################
#     Tests for FlattenModule     #
###################################


def test_flatten_module_forward_args():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert FlattenModule(module)(torch.ones(6, 1), torch.zeros(6, 1)).equal(torch.tensor(1.0))
    assert objects_are_equal(module.call_args.args, (torch.ones(6), torch.zeros(6)))


def test_flatten_module_forward_kwargs():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert FlattenModule(module)(input1=torch.ones(6, 1), input2=torch.zeros(6, 1)).equal(
        torch.tensor(1.0)
    )
    assert objects_are_equal(
        module.call_args.kwargs, {"input1": torch.ones(6), "input2": torch.zeros(6)}
    )


def test_flatten_module_forward_args_and_kwargs():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert FlattenModule(module)(torch.ones(6, 1), input2=torch.zeros(6, 1)).equal(
        torch.tensor(1.0)
    )
    assert objects_are_equal(module.call_args.args, (torch.ones(6),))
    assert objects_are_equal(module.call_args.kwargs, {"input2": torch.zeros(6)})


def test_flatten_module_mse():
    module = FlattenModule(nn.MSELoss(reduction="sum"))
    assert module(torch.ones(6, 1), torch.zeros(6, 1)).equal(torch.tensor(6.0))


#############################################
#     Tests for MulticlassFlattenModule     #
#############################################


def test_flatten_multiclass_module_forward_args():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert MulticlassFlattenModule(module)(torch.ones(6, 2, 4), torch.zeros(6, 2)).equal(
        torch.tensor(1.0)
    )
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_module_forward_kwargs():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert MulticlassFlattenModule(module)(
        prediction=torch.ones(6, 2, 4), target=torch.zeros(6, 2)
    ).equal(torch.tensor(1.0))
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_module_forward_args_and_kwargs():
    module = Mock(spec=torch.nn.Module, return_value=torch.tensor(1.0))
    assert MulticlassFlattenModule(module)(torch.ones(6, 2, 4), target=torch.zeros(6, 2)).equal(
        torch.tensor(1.0)
    )
    assert objects_are_equal(module.call_args.args, (torch.ones(12, 4), torch.zeros(12)))


def test_flatten_multiclass_module_cross_entropy():
    module = MulticlassFlattenModule(nn.CrossEntropyLoss())
    assert module(torch.ones(6, 2, 4), torch.zeros(6, 2, dtype=torch.long)).allclose(
        torch.tensor(1.3862943649291992),
        atol=1e-6,
    )
