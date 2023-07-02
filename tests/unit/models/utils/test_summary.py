from __future__ import annotations

import copy

import torch
from pytest import raises
from torch import Tensor, nn

from gravitorch.models.networks import BetaMLP
from gravitorch.models.utils.summary import ModelSummary, model_forward_dummy_input
from gravitorch.nn import ConcatFusion
from gravitorch.nn.utils.summary import UNKNOWN_DTYPE, UNKNOWN_SIZE, ModuleSummary

SIZES = (1, 2)

LINEAR_TABLE_STR = (
    "╒════╤═══════════════╤════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕\n"  # noqa: E501,B950
    "│    │ Name          │ Type   │   Params │   Learn Params │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │\n"  # noqa: E501,B950
    "╞════╪═══════════════╪════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡\n"  # noqa: E501,B950
    "│  0 │ [root module] │ Linear │       25 │             25 │ (1, 4)     │ torch.float32 │ (1, 5)      │ torch.float32 │\n"  # noqa: E501,B950
    "╘════╧═══════════════╧════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛\n"  # noqa: E501,B950
    " - 25         Learnable params\n"
    " - 0          Non-learnable params\n"
    " - 25         Total params\n"
)

MLP_TABLE_TOP_STR = (
    "╒════╤═══════════════╤════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕\n"  # noqa: E501,B950
    "│    │ Name          │ Type       │ Params   │ Learn Params   │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │\n"  # noqa: E501,B950
    "╞════╪═══════════════╪════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡\n"  # noqa: E501,B950
    "│  0 │ [root module] │ BetaMLP    │ 27.2 K   │ 27.2 K         │ (1, 32)    │ torch.float32 │ (1, 50)     │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼───────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  1 │ layers        │ Sequential │ 27.2 K   │ 27.2 K         │ (1, 32)    │ torch.float32 │ (1, 50)     │ torch.float32 │\n"  # noqa: E501,B950
    "╘════╧═══════════════╧════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛\n"  # noqa: E501,B950
    " - 27.2 K     Learnable params\n"
    " - 0          Non-learnable params\n"
    " - 27.2 K     Total params\n"
)

MLP_TABLE_FULL_STR = (
    "╒════╤════════════════╤════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕\n"  # noqa: E501,B950
    "│    │ Name           │ Type       │ Params   │ Learn Params   │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │\n"  # noqa: E501,B950
    "╞════╪════════════════╪════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡\n"  # noqa: E501,B950
    "│  0 │ [root module]  │ BetaMLP    │ 27.2 K   │ 27.2 K         │ (1, 32)    │ torch.float32 │ (1, 50)     │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  1 │ layers         │ Sequential │ 27.2 K   │ 27.2 K         │ (1, 32)    │ torch.float32 │ (1, 50)     │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  2 │ layers.linear1 │ Linear     │ 4.2 K    │ 4.2 K          │ (1, 32)    │ torch.float32 │ (1, 128)    │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  3 │ layers.relu1   │ ReLU       │ 0        │ 0              │ (1, 128)   │ torch.float32 │ (1, 128)    │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  4 │ layers.linear2 │ Linear     │ 16.5 K   │ 16.5 K         │ (1, 128)   │ torch.float32 │ (1, 128)    │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  5 │ layers.relu2   │ ReLU       │ 0        │ 0              │ (1, 128)   │ torch.float32 │ (1, 128)    │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────┼────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  6 │ layers.linear3 │ Linear     │ 6.5 K    │ 6.5 K          │ (1, 128)   │ torch.float32 │ (1, 50)     │ torch.float32 │\n"  # noqa: E501,B950
    "╘════╧════════════════╧════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛\n"  # noqa: E501,B950
    " - 27.2 K     Learnable params\n"
    " - 0          Non-learnable params\n"
    " - 27.2 K     Total params\n"
)

MY_NETWORK_TABLE_TOP_STR = (
    "╒════╤═══════════════╤══════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕\n"  # noqa: E501,B950
    "│    │ Name          │ Type         │   Params │   Learn Params │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │\n"  # noqa: E501,B950
    "╞════╪═══════════════╪══════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡\n"  # noqa: E501,B950
    "│  0 │ [root module] │ MyNetwork    │      455 │            455 │ (1, 10)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "│    │               │              │          │                │ (1, 2)     │ torch.float32 │ (1, 8)      │ torch.float32 │\n"  # noqa: E501,B950
    "│    │               │              │          │                │ (1, 8)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "├────┼───────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  1 │ fusion        │ ConcatFusion │        0 │              0 │ (1, 10)    │ torch.float32 │ (1, 20)     │ torch.float32 │\n"  # noqa: E501,B950
    "│    │               │              │          │                │ (1, 2)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "│    │               │              │          │                │ (1, 8)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "├────┼───────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  2 │ decoder       │ BetaMLP      │      455 │            455 │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "╘════╧═══════════════╧══════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛\n"  # noqa: E501,B950
    " - 455        Learnable params\n"
    " - 0          Non-learnable params\n"
    " - 455        Total params\n"
)

MY_NETWORK_TABLE_FULL_STR = (
    "╒════╤════════════════════════╤══════════════╤══════════╤════════════════╤════════════╤═══════════════╤═════════════╤═══════════════╕\n"  # noqa: E501,B950
    "│    │ Name                   │ Type         │   Params │   Learn Params │ In sizes   │ In dtype      │ Out sizes   │ Out dtype     │\n"  # noqa: E501,B950
    "╞════╪════════════════════════╪══════════════╪══════════╪════════════════╪════════════╪═══════════════╪═════════════╪═══════════════╡\n"  # noqa: E501,B950
    "│  0 │ [root module]          │ MyNetwork    │      455 │            455 │ (1, 10)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "│    │                        │              │          │                │ (1, 2)     │ torch.float32 │ (1, 8)      │ torch.float32 │\n"  # noqa: E501,B950
    "│    │                        │              │          │                │ (1, 8)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  1 │ fusion                 │ ConcatFusion │        0 │              0 │ (1, 10)    │ torch.float32 │ (1, 20)     │ torch.float32 │\n"  # noqa: E501,B950
    "│    │                        │              │          │                │ (1, 2)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "│    │                        │              │          │                │ (1, 8)     │ torch.float32 │             │               │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  2 │ decoder                │ BetaMLP      │      455 │            455 │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  3 │ decoder.layers         │ Sequential   │      455 │            455 │ (1, 20)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  4 │ decoder.layers.linear1 │ Linear       │      336 │            336 │ (1, 20)    │ torch.float32 │ (1, 16)     │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  5 │ decoder.layers.relu1   │ ReLU         │        0 │              0 │ (1, 16)    │ torch.float32 │ (1, 16)     │ torch.float32 │\n"  # noqa: E501,B950
    "├────┼────────────────────────┼──────────────┼──────────┼────────────────┼────────────┼───────────────┼─────────────┼───────────────┤\n"  # noqa: E501,B950
    "│  6 │ decoder.layers.linear2 │ Linear       │      119 │            119 │ (1, 16)    │ torch.float32 │ (1, 7)      │ torch.float32 │\n"  # noqa: E501,B950
    "╘════╧════════════════════════╧══════════════╧══════════╧════════════════╧════════════╧═══════════════╧═════════════╧═══════════════╛\n"  # noqa: E501,B950
    " - 455        Learnable params\n"
    " - 0          Non-learnable params\n"
    " - 455        Total params\n"
)


class MyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = ConcatFusion()
        self.decoder = BetaMLP(input_size=20, hidden_sizes=(16, 7))

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> tuple[Tensor, Tensor]:
        return self.decoder(self.fusion(x1, x2, x3)), x3

    def get_dummy_input(self, batch_size: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.randn(batch_size, 10),
            torch.randn(batch_size, 2),
            torch.randn(batch_size, 8),
        )


##################################
#     Tests for ModelSummary     #
##################################


def test_model_summary_linear_invalid_mode() -> None:
    module = nn.Linear(4, 5)
    with raises(ValueError, match="Incorrect mode: invalid mode."):
        ModelSummary(module, mode="invalid mode")


def test_model_summary_linear_top() -> None:
    module = nn.Linear(4, 5)
    summary = ModelSummary(module, mode=ModelSummary.MODE_TOP)
    assert summary.layer_names == ("[root module]",)
    assert summary.layer_types == ("Linear",)
    assert summary.param_nums == (25,)
    assert summary.learn_param_nums == (25,)
    assert summary.in_sizes == (UNKNOWN_SIZE,)
    assert summary.out_sizes == (UNKNOWN_SIZE,)
    assert summary.in_dtypes == (UNKNOWN_DTYPE,)
    assert summary.out_dtypes == (UNKNOWN_DTYPE,)
    assert str(summary) == (
        "╒════╤═══════════════╤════════╤══════════╤════════════════╕\n"  # noqa: E501,B950
        "│    │ Name          │ Type   │   Params │   Learn Params │\n"  # noqa: E501,B950
        "╞════╪═══════════════╪════════╪══════════╪════════════════╡\n"  # noqa: E501,B950
        "│  0 │ [root module] │ Linear │       25 │             25 │\n"  # noqa: E501,B950
        "╘════╧═══════════════╧════════╧══════════╧════════════════╛\n"  # noqa: E501,B950
        " - 25         Learnable params\n"
        " - 0          Non-learnable params\n"
        " - 25         Total params\n"
    )


def test_model_summary_linear_with_get_dummy_input_top() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: torch.randn(1, 4)
    summary = ModelSummary(module, mode=ModelSummary.MODE_TOP)
    assert summary.layer_names == ("[root module]",)
    assert summary.layer_types == ("Linear",)
    assert summary.param_nums == (25,)
    assert summary.learn_param_nums == (25,)
    assert summary.in_sizes == ((1, 4),)
    assert summary.out_sizes == ((1, 5),)
    assert summary.in_dtypes == ("torch.float32",)
    assert summary.out_dtypes == ("torch.float32",)
    assert str(summary) == LINEAR_TABLE_STR


def test_model_summary_linear_full() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: torch.randn(1, 4)
    summary = ModelSummary(module, mode=ModelSummary.MODE_FULL)
    assert summary.layer_names == ("[root module]",)
    assert summary.layer_types == ("Linear",)
    assert summary.param_nums == (25,)
    assert summary.learn_param_nums == (25,)
    assert summary.in_sizes == ((1, 4),)
    assert summary.out_sizes == ((1, 5),)
    assert summary.in_dtypes == ("torch.float32",)
    assert summary.out_dtypes == ("torch.float32",)
    assert str(summary) == LINEAR_TABLE_STR


def test_model_summary_linear_get_dummy_input_tuple() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: (torch.randn(1, 4),)
    summary = ModelSummary(module, mode=ModelSummary.MODE_FULL)
    assert summary.layer_names == ("[root module]",)
    assert summary.layer_types == ("Linear",)
    assert summary.param_nums == (25,)
    assert summary.learn_param_nums == (25,)
    assert summary.in_sizes == ((1, 4),)
    assert summary.out_sizes == ((1, 5),)
    assert summary.in_dtypes == ("torch.float32",)
    assert summary.out_dtypes == ("torch.float32",)
    assert str(summary) == LINEAR_TABLE_STR


def test_model_summary_linear_get_dummy_input_list() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: (torch.randn(1, 4),)
    summary = ModelSummary(module, mode=ModelSummary.MODE_FULL)
    assert summary.layer_names == ("[root module]",)
    assert summary.layer_types == ("Linear",)
    assert summary.param_nums == (25,)
    assert summary.learn_param_nums == (25,)
    assert summary.in_sizes == ((1, 4),)
    assert summary.out_sizes == ((1, 5),)
    assert summary.in_dtypes == ("torch.float32",)
    assert summary.out_dtypes == ("torch.float32",)
    assert str(summary) == LINEAR_TABLE_STR


def test_model_summary_mlp_top() -> None:
    module = BetaMLP(input_size=32, hidden_sizes=(128, 128, 50))
    summary = ModelSummary(module, mode=ModelSummary.MODE_TOP)
    assert summary.layer_names == ("[root module]", "layers")
    assert summary.layer_types == ("BetaMLP", "Sequential")
    assert summary.param_nums == (27186, 27186)  # (32 + 1) * 128 + (128 + 1) * 128 + (128 + 1) * 50
    assert summary.learn_param_nums == (27186, 27186)
    assert summary.in_sizes == ((1, 32), (1, 32))
    assert summary.out_sizes == ((1, 50), (1, 50))
    assert summary.in_dtypes == ("torch.float32", "torch.float32")
    assert summary.out_dtypes == ("torch.float32", "torch.float32")
    assert str(summary) == MLP_TABLE_TOP_STR


def test_model_summary_mlp_full() -> None:
    module = BetaMLP(input_size=32, hidden_sizes=(128, 128, 50))
    summary = ModelSummary(module, mode=ModelSummary.MODE_FULL)
    assert summary.layer_names == (
        "[root module]",
        "layers",
        "layers.linear1",
        "layers.relu1",
        "layers.linear2",
        "layers.relu2",
        "layers.linear3",
    )
    assert summary.layer_types == (
        "BetaMLP",
        "Sequential",
        "Linear",
        "ReLU",
        "Linear",
        "ReLU",
        "Linear",
    )
    assert summary.param_nums == (27186, 27186, 4224, 0, 16512, 0, 6450)
    assert summary.learn_param_nums == (27186, 27186, 4224, 0, 16512, 0, 6450)
    assert summary.in_sizes == ((1, 32), (1, 32), (1, 32), (1, 128), (1, 128), (1, 128), (1, 128))
    assert summary.out_sizes == ((1, 50), (1, 50), (1, 128), (1, 128), (1, 128), (1, 128), (1, 50))
    assert summary.in_dtypes == (
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
    )
    assert summary.out_dtypes == (
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
    )
    assert str(summary) == MLP_TABLE_FULL_STR


def test_model_summary_my_network_top() -> None:
    module = MyNetwork()
    summary = ModelSummary(module, mode=ModelSummary.MODE_TOP)
    assert summary.layer_names == ("[root module]", "fusion", "decoder")
    assert summary.layer_types == ("MyNetwork", "ConcatFusion", "BetaMLP")
    assert summary.param_nums == (455, 0, 455)
    assert summary.learn_param_nums == (455, 0, 455)
    assert summary.in_sizes == (((1, 10), (1, 2), (1, 8)), ((1, 10), (1, 2), (1, 8)), (1, 20))
    assert summary.out_sizes == (((1, 7), (1, 8)), (1, 20), (1, 7))
    assert summary.in_dtypes == (
        ("torch.float32", "torch.float32", "torch.float32"),
        ("torch.float32", "torch.float32", "torch.float32"),
        "torch.float32",
    )
    assert summary.out_dtypes == (
        ("torch.float32", "torch.float32"),
        "torch.float32",
        "torch.float32",
    )
    assert str(summary) == MY_NETWORK_TABLE_TOP_STR


def test_model_summary_my_network_full() -> None:
    module = MyNetwork()
    summary = ModelSummary(module, mode=ModelSummary.MODE_FULL)
    assert summary.layer_names == (
        "[root module]",
        "fusion",
        "decoder",
        "decoder.layers",
        "decoder.layers.linear1",
        "decoder.layers.relu1",
        "decoder.layers.linear2",
    )
    assert summary.layer_types == (
        "MyNetwork",
        "ConcatFusion",
        "BetaMLP",
        "Sequential",
        "Linear",
        "ReLU",
        "Linear",
    )
    assert summary.param_nums == (455, 0, 455, 455, 336, 0, 119)
    assert summary.learn_param_nums == (455, 0, 455, 455, 336, 0, 119)
    assert summary.in_sizes == (
        ((1, 10), (1, 2), (1, 8)),
        ((1, 10), (1, 2), (1, 8)),
        (1, 20),
        (1, 20),
        (1, 20),
        (1, 16),
        (1, 16),
    )
    assert summary.out_sizes == (
        ((1, 7), (1, 8)),
        (1, 20),
        (1, 7),
        (1, 7),
        (1, 16),
        (1, 16),
        (1, 7),
    )
    assert summary.in_dtypes == (
        ("torch.float32", "torch.float32", "torch.float32"),
        ("torch.float32", "torch.float32", "torch.float32"),
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
    )
    assert summary.out_dtypes == (
        ("torch.float32", "torch.float32"),
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
        "torch.float32",
    )
    assert str(summary) == MY_NETWORK_TABLE_FULL_STR


###############################################
#     Tests for model_forward_dummy_input     #
###############################################


def test_model_forward_dummy_input_without_get_dummy_input() -> None:
    module = nn.Linear(4, 6)
    summary = ModuleSummary(module)
    model_forward_dummy_input(module)
    # The hook should not be removed because the forward function was not computed
    assert summary._hook_handle.id in module._forward_hooks


def test_model_forward_dummy_input_with_get_dummy_input_tensor() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: torch.randn(1, 4)
    summary = ModuleSummary(module)
    model_forward_dummy_input(module)
    # The hook should be removed because the forward function was computed
    assert summary._hook_handle.id not in module._forward_hooks


def test_model_forward_dummy_input_with_get_dummy_input_tuple() -> None:
    module = copy.deepcopy(nn.Linear(4, 5))
    module.get_dummy_input = lambda *args, **kwargs: (torch.randn(1, 4),)
    summary = ModuleSummary(module)
    model_forward_dummy_input(module)
    # The hook should be removed because the forward function was computed
    assert summary._hook_handle.id not in module._forward_hooks
