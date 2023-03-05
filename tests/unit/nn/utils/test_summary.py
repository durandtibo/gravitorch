import torch
from pytest import mark
from torch import Tensor, nn

from gravitorch.nn import freeze_module
from gravitorch.nn.utils.summary import (
    UNKNOWN_DTYPE,
    UNKNOWN_SIZE,
    ModuleSummary,
    multiline_format_dtype,
    multiline_format_size,
    parse_batch_dtype,
    parse_batch_shape,
)
from gravitorch.utils import get_available_devices

SIZES = (1, 2)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=8)
        # The parameters of the embedding layer should not appear in the learnable parameters.
        freeze_module(self.embedding)
        self.fc = nn.Linear(8, 4)

    def forward(self, tensor: Tensor) -> Tensor:
        return self.fc(self.embedding(tensor))

    def get_dummy_input(self, batch_size: int = 1):
        return (torch.ones(batch_size, dtype=torch.long),)


###################################
#     Tests for ModuleSummary     #
###################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_module_summary_linear(device: str, batch_size: int, input_size: int, output_size: int):
    device = torch.device(device)
    module = nn.Linear(input_size, output_size).to(device=device)
    summary = ModuleSummary(module)
    assert summary.num_parameters == input_size * output_size + output_size
    assert summary.num_learnable_parameters == input_size * output_size + output_size
    assert summary.layer_type == "Linear"
    # Run the forward to capture the input and output sizes.
    module(torch.rand(batch_size, input_size, device=device))
    assert summary.in_size == (batch_size, input_size)
    assert summary.out_size == (batch_size, output_size)
    assert summary.in_dtype == "torch.float32"
    assert summary.out_dtype == "torch.float32"


@mark.parametrize("device", get_available_devices())
def test_module_summary_bilinear(device: str):
    device = torch.device(device)
    module = nn.Bilinear(in1_features=3, in2_features=4, out_features=7).to(device=device)
    summary = ModuleSummary(module)
    print(summary.num_parameters)
    assert summary.num_parameters == 91
    assert summary.num_learnable_parameters == 91
    assert summary.layer_type == "Bilinear"
    # Run the forward to capture the input and output sizes.
    module(torch.rand(2, 3, device=device), torch.rand(2, 4, device=device))
    assert summary.in_size == ((2, 3), (2, 4))
    assert summary.out_size == (2, 7)
    assert summary.in_dtype == ("torch.float32", "torch.float32")
    assert summary.out_dtype == "torch.float32"


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("output_size", SIZES)
def test_module_summary_linear_no_forward(
    device: str, batch_size: int, input_size: int, output_size: int
):
    device = torch.device(device)
    module = nn.Linear(input_size, output_size).to(device=device)
    summary = ModuleSummary(module)
    assert summary.num_parameters == input_size * output_size + output_size
    assert summary.num_learnable_parameters == input_size * output_size + output_size
    assert summary.layer_type == "Linear"
    assert summary.in_size == UNKNOWN_SIZE
    assert summary.out_size == UNKNOWN_SIZE
    assert summary.in_dtype == UNKNOWN_DTYPE
    assert summary.out_dtype == UNKNOWN_DTYPE


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
@mark.parametrize("input_size", SIZES)
@mark.parametrize("hidden_size", SIZES)
def test_module_summary_gru(
    device: str, batch_size: int, seq_len: int, input_size: int, hidden_size: int
):
    device = torch.device(device)
    module = nn.GRU(input_size, hidden_size).to(device=device)
    summary = ModuleSummary(module)
    num_parameters = 3 * ((input_size + 1) * hidden_size + (hidden_size + 1) * hidden_size)
    assert summary.num_parameters == num_parameters
    assert summary.num_learnable_parameters == num_parameters
    assert summary.layer_type == "GRU"
    # Run the forward to capture the input and output sizes.
    module(torch.rand(seq_len, batch_size, input_size, device=device))
    assert summary.in_size == (seq_len, batch_size, input_size)
    assert summary.out_size == ((seq_len, batch_size, hidden_size), (1, batch_size, hidden_size))
    assert summary.in_dtype == "torch.float32"
    assert summary.out_dtype == ("torch.float32", "torch.float32")


@mark.parametrize("device", get_available_devices())
@mark.parametrize("batch_size", SIZES)
def test_module_summary_custom_module(device: str, batch_size: int):
    device = torch.device(device)
    module = MyModule().to(device=device)
    summary = ModuleSummary(module)
    assert summary.num_parameters == 116
    assert summary.num_learnable_parameters == 36
    assert summary.layer_type == "MyModule"
    # Run the forward to capture the input and output sizes.
    module(*[dummy_input.to(device=device) for dummy_input in module.get_dummy_input(batch_size)])
    assert summary.in_size == (batch_size,)
    assert summary.out_size == (batch_size, 4)
    assert summary.in_dtype == "torch.int64"
    assert summary.out_dtype == "torch.float32"


def test_module_summary_detach():
    module = nn.Linear(4, 6)
    summary = ModuleSummary(module)
    assert summary._hook_handle.id in module._forward_hooks
    summary.detach_hook()
    assert summary._hook_handle.id not in module._forward_hooks


#######################################
#     Tests for parse_batch_shape     #
#######################################


@mark.parametrize("batch_size", SIZES)
@mark.parametrize("input_size", SIZES)
def test_parse_batch_shape_tensor_2d(batch_size: int, input_size: int):
    assert parse_batch_shape(torch.ones(batch_size, input_size)) == (batch_size, input_size)


@mark.parametrize("seq_len,batch_size,input_size", ((1, 1, 1), (2, 2, 2)))
def test_parse_batch_shape_tensor_3d(seq_len: int, batch_size: int, input_size: int):
    assert parse_batch_shape(torch.ones(seq_len, batch_size, input_size)) == (
        seq_len,
        batch_size,
        input_size,
    )


@mark.parametrize(
    "batch",
    [
        (torch.ones(2, 3), torch.ones(2, dtype=torch.long)),
        [torch.ones(2, 3), torch.ones(2, dtype=torch.long)],
    ],
)
def test_parse_batch_shape_list_tuple(batch: Tensor):
    assert parse_batch_shape(batch) == ((2, 3), (2,))


def test_parse_batch_shape_dict():
    assert (
        parse_batch_shape(
            {"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)}
        )
        == UNKNOWN_SIZE
    )


#######################################
#     Tests for parse_batch_dtype     #
#######################################


@mark.parametrize(
    "dtype,parsed_dtype",
    (
        (torch.bool, "torch.bool"),
        (torch.uint8, "torch.uint8"),
        (torch.int, "torch.int32"),
        (torch.int16, "torch.int16"),
        (torch.int32, "torch.int32"),
        (torch.int64, "torch.int64"),
        (torch.long, "torch.int64"),
        (torch.float, "torch.float32"),
        (torch.float16, "torch.float16"),
        (torch.float32, "torch.float32"),
        (torch.float64, "torch.float64"),
        (torch.cdouble, "torch.complex128"),
        (torch.complex64, "torch.complex64"),
    ),
)
def test_parse_batch_dtype_tensor(dtype: torch.dtype, parsed_dtype: str):
    assert parse_batch_dtype(torch.ones(2, 3, dtype=dtype)) == parsed_dtype


@mark.parametrize(
    "batch",
    [
        (torch.ones(2, 3), torch.ones(2, dtype=torch.long)),
        [torch.ones(2, 3), torch.ones(2, dtype=torch.long)],
    ],
)
def test_parse_batch_dtype_list_tuple(batch):
    assert parse_batch_dtype(batch) == ("torch.float32", "torch.int64")


def test_parse_batch_dtype_nested_tuple():
    assert parse_batch_dtype(
        (torch.ones(2, 3), (torch.ones(2, dtype=torch.long), torch.ones(2, 3)))
    ) == (
        "torch.float32",
        ("torch.int64", "torch.float32"),
    )


def test_parse_batch_dtype_dict():
    batch = {"feature1": torch.ones(2, 3), "feature2": torch.ones(2, dtype=torch.long)}
    assert parse_batch_dtype(batch) == UNKNOWN_DTYPE


###########################################
#     Tests for multiline_format_size     #
###########################################


def test_multiline_format_size_list():
    assert multiline_format_size([UNKNOWN_SIZE, [1, 4], [[2, 4], [2]]]) == (
        UNKNOWN_SIZE,
        "[1, 4]",
        "[2, 4]\n[2]",
    )


def test_multiline_format_size_tuple():
    assert multiline_format_size((UNKNOWN_SIZE, (1, 4), ((2, 4), (2,)))) == (
        UNKNOWN_SIZE,
        "(1, 4)",
        "(2, 4)\n(2,)",
    )


############################################
#     Tests for multiline_format_dtype     #
############################################


def test_multiline_format_dtype_list():
    assert multiline_format_dtype(
        [UNKNOWN_DTYPE, "torch.float32", ["torch.float32", "torch.int32"]]
    ) == (
        UNKNOWN_SIZE,
        "torch.float32",
        "torch.float32\ntorch.int32",
    )


def test_multiline_format_dtype_tuple():
    assert multiline_format_dtype(
        (UNKNOWN_DTYPE, "torch.float32", ("torch.float32", "torch.int32"))
    ) == (
        UNKNOWN_SIZE,
        "torch.float32",
        "torch.float32\ntorch.int32",
    )
