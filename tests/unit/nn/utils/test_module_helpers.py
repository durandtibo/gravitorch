import torch
from pytest import mark, raises
from torch import nn

from gravitorch.nn import (
    freeze_module,
    get_module_device,
    get_module_devices,
    get_module_input_size,
    get_module_output_size,
    has_batch_norm,
    has_learnable_parameters,
    has_parameters,
    is_batch_first,
    is_module_on_device,
    num_learnable_parameters,
    num_parameters,
    unfreeze_module,
)
from gravitorch.nn.utils.module_helpers import (
    _get_sequential_input_size,
    _get_sequential_output_size,
    get_module_name,
    module_mode,
    top_module_mode,
)
from tests.testing import cuda_available

SIZES = (1, 2)


####################################
#     Tests for has_parameters     #
####################################


def test_has_parameters_true():
    assert has_parameters(nn.Linear(4, 5))


def test_has_parameters_false():
    assert not has_parameters(nn.Tanh())


##############################################
#     Tests for has_learnable_parameters     #
##############################################


def test_has_learnable_parameters_true():
    assert has_learnable_parameters(nn.Linear(4, 5))


def test_has_learnable_parameters_false():
    assert not has_learnable_parameters(nn.Tanh())


####################################
#     Tests for num_parameters     #
####################################


def test_num_parameters_0():
    assert num_parameters(nn.Tanh()) == 0


def test_num_parameters_15():
    assert num_parameters(nn.Linear(2, 5)) == 15  # 10 (weight) + 5 (bias)


def test_num_parameters_25():
    assert num_parameters(nn.Linear(4, 5)) == 25  # 20 (weight) + 5 (bias)


def test_num_parameters_25_frozen():
    module = nn.Linear(4, 5)
    freeze_module(module)
    assert num_parameters(module) == 25  # 20 (weight) + 5 (bias)


def test_num_parameters_2_layers():
    fc1 = nn.Linear(4, 5)
    fc2 = nn.Linear(5, 8)
    model = nn.Sequential(fc1, fc2)
    assert num_parameters(model) == 25 + 48
    # Freeze the parameters of FC2.
    freeze_module(fc2)
    assert num_parameters(model) == 25 + 48


##############################################
#     Tests for num_learnable_parameters     #
##############################################


def test_num_learnable_parameters_0():
    assert num_learnable_parameters(nn.Tanh()) == 0


def test_num_learnable_parameters_15():
    assert num_learnable_parameters(nn.Linear(2, 5)) == 15  # 10 (weight) + 5 (bias)


def test_num_learnable_parameters_25():
    assert num_learnable_parameters(nn.Linear(4, 5)) == 25  # 20 (weight) + 5 (bias)


def test_num_learnable_parameters_2_layers():
    fc1 = nn.Linear(4, 5)
    fc2 = nn.Linear(5, 8)
    model = nn.Sequential(fc1, fc2)
    assert num_learnable_parameters(model) == 25 + 48
    # Freeze the parameters of FC2.
    freeze_module(fc2)
    assert num_learnable_parameters(model) == 25  # 20 (weight) + 5 (bias)


@mark.parametrize("module", [nn.Tanh(), nn.Linear(2, 5), nn.Linear(4, 5)])
def test_freeze_module(module: nn.Module):
    freeze_module(module)
    assert num_learnable_parameters(module) == 0


def test_unfreeze_module_tanh():
    module = nn.Tanh()
    freeze_module(module)
    assert num_learnable_parameters(module) == 0
    unfreeze_module(module)
    assert num_learnable_parameters(module) == 0


def test_unfreeze_module_linear():
    module = nn.Linear(4, 5)
    freeze_module(module)
    assert num_learnable_parameters(module) == 0
    unfreeze_module(module)
    assert num_learnable_parameters(module) == 25  # 20 (weight) + 5 (bias)


#######################################
#     Tests for get_module_device     #
#######################################


def test_get_module_device_cpu():
    assert get_module_device(nn.Linear(4, 5)) == torch.device("cpu")


@cuda_available
def test_get_module_device_cuda():
    assert get_module_device(nn.Linear(4, 5).to(device=torch.device("cuda:0"))) == torch.device(
        "cuda:0"
    )


def test_get_module_device_no_parameter():
    assert get_module_device(nn.Identity()) == torch.device("cpu")


def test_get_module_devices_cpu():
    assert get_module_devices(nn.Linear(4, 5)) == (torch.device("cpu"),)


@cuda_available
def test_get_module_devices_cuda():
    assert get_module_devices(nn.Linear(4, 5).to(device=torch.device("cuda:0"))) == (
        torch.device("cuda:0"),
    )


@cuda_available
def test_get_module_devices_cpu_cuda():
    net = nn.Sequential(nn.Linear(4, 5), nn.Linear(4, 5).to(device=torch.device("cuda:0")))
    assert set(get_module_devices(net)) == {torch.device("cpu"), torch.device("cuda:0")}


#######################################
#     Tests for get_module_device     #
#######################################


def test_is_module_on_device_true():
    assert is_module_on_device(nn.Linear(4, 5), torch.device("cpu"))


def test_is_module_on_device_false():
    assert not is_module_on_device(nn.Linear(4, 5), torch.device("cuda:0"))


###########################################
#     Tests for get_module_input_size     #
###########################################


@mark.parametrize(
    "module",
    (
        nn.RNN(input_size=6, hidden_size=4),
        nn.LSTM(input_size=6, hidden_size=4),
        nn.GRU(input_size=6, hidden_size=4),
    ),
)
def test_get_module_input_size_module_with_input_size(module: nn.Module):
    assert get_module_input_size(module) == 6


@mark.parametrize(
    "module",
    (
        nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3),
        nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3),
        nn.Conv3d(in_channels=6, out_channels=4, kernel_size=3),
    ),
)
def test_get_module_input_size_module_with_in_channels(module: nn.Module):
    assert get_module_input_size(module) == 6


@mark.parametrize("input_size", SIZES)
def test_get_module_input_size_linear(input_size: int):
    assert get_module_input_size(nn.Linear(input_size, 2)) == input_size


@mark.parametrize("input_size", SIZES)
def test_get_module_input_size_sequential(input_size: int):
    assert get_module_input_size(nn.Sequential(nn.Linear(input_size, 2), nn.ReLU())) == input_size


@mark.parametrize("input_size", SIZES)
def test_get_module_input_size_module_list(input_size: int):
    assert (
        get_module_input_size(nn.ModuleList([nn.Linear(input_size, 10) for _ in range(5)]))
        == input_size
    )


@mark.parametrize("input_size", SIZES)
def test_get_module_input_size_multihead_attention(input_size: int):
    assert get_module_input_size(nn.MultiheadAttention(input_size, 1)) == input_size


@mark.parametrize(
    "module",
    (
        nn.TransformerEncoderLayer(
            d_model=6,
            nhead=1,
            dim_feedforward=24,
            dropout=0.1,
            activation="gelu",
        ),
        nn.TransformerDecoderLayer(
            d_model=6,
            nhead=1,
            dim_feedforward=24,
            dropout=0.1,
            activation="gelu",
        ),
        nn.TransformerEncoder(
            num_layers=2,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=6,
                nhead=1,
                dim_feedforward=24,
                dropout=0.1,
                activation="gelu",
            ),
        ),
        nn.TransformerDecoder(
            num_layers=2,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=6,
                nhead=1,
                dim_feedforward=24,
                dropout=0.1,
                activation="gelu",
            ),
        ),
    ),
)
def test_get_module_input_size_transformer(module: nn.Module):
    assert get_module_input_size(module) == 6


def test_get_module_input_size_incorrect_type():
    with raises(TypeError):
        get_module_input_size(nn.Dropout(0.5))


################################################
#     Tests for _get_sequential_input_size     #
################################################


@mark.parametrize("input_size", SIZES)
def test_get_sequential_input_size_first(input_size: int):
    assert (
        _get_sequential_input_size(nn.Sequential(nn.Linear(input_size, 4), nn.ReLU())) == input_size
    )


@mark.parametrize("input_size", SIZES)
def test_get_sequential_input_size_non_first(input_size: int):
    assert (
        _get_sequential_input_size(nn.Sequential(nn.ReLU(), nn.Linear(input_size, 6))) == input_size
    )


def test_get_sequential_input_size_error():
    with raises(TypeError):
        _get_sequential_input_size(nn.Sequential(nn.Dropout(0.5), nn.ReLU()))


def test_get_sequential_input_size_incorrect_input():
    with raises(TypeError):
        _get_sequential_input_size(nn.Dropout(0.5))


############################################
#     Tests for get_module_output_size     #
############################################


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_output_size(output_size: int):
    assert get_module_output_size(nn.AdaptiveAvgPool1d(output_size)) == output_size


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_out_features(output_size: int):
    assert get_module_output_size(nn.Linear(4, output_size)) == output_size


@mark.parametrize(
    "module",
    (
        nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3),
        nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3),
        nn.Conv3d(in_channels=6, out_channels=4, kernel_size=3),
    ),
)
def test_get_module_output_size_module_with_out_channels(module: nn.Module):
    assert get_module_output_size(module) == 4


@mark.parametrize(
    "module",
    (
        nn.RNN(input_size=6, hidden_size=4),
        nn.LSTM(input_size=6, hidden_size=4),
        nn.GRU(input_size=6, hidden_size=4),
    ),
)
def test_get_module_output_size_module_recurrent(module: nn.Module):
    assert get_module_output_size(module) == 4


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_embedding(output_size: int):
    assert (
        get_module_output_size(nn.Embedding(num_embeddings=4, embedding_dim=output_size))
        == output_size
    )


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_sequential(output_size: int):
    assert (
        get_module_output_size(nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, output_size)))
        == output_size
    )


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_module_list(output_size: int):
    assert (
        get_module_output_size(nn.ModuleList([nn.Linear(10, output_size) for _ in range(5)]))
        == output_size
    )


@mark.parametrize("output_size", SIZES)
def test_get_module_output_size_multihead_attention(output_size: int):
    assert get_module_output_size(nn.MultiheadAttention(output_size, 1)) == output_size


@mark.parametrize(
    "module",
    (
        nn.TransformerEncoderLayer(
            d_model=6,
            nhead=1,
            dim_feedforward=24,
            dropout=0.1,
            activation="gelu",
        ),
        nn.TransformerDecoderLayer(
            d_model=6,
            nhead=1,
            dim_feedforward=24,
            dropout=0.1,
            activation="gelu",
        ),
        nn.TransformerEncoder(
            num_layers=2,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=6,
                nhead=1,
                dim_feedforward=24,
                dropout=0.1,
                activation="gelu",
            ),
        ),
        nn.TransformerDecoder(
            num_layers=2,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=6,
                nhead=1,
                dim_feedforward=24,
                dropout=0.1,
                activation="gelu",
            ),
        ),
    ),
)
def test_get_module_output_size_transformer(module: nn.Module):
    assert get_module_output_size(module) == 6


def test_get_module_output_size_incorrect_type():
    with raises(TypeError):
        get_module_output_size(nn.Dropout(0.5))


#################################################
#     Tests for _get_sequential_output_size     #
#################################################


@mark.parametrize("output_size", SIZES)
def test_get_sequential_output_size_last(output_size: int):
    assert (
        _get_sequential_output_size(
            nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, output_size))
        )
        == output_size
    )


@mark.parametrize("output_size", SIZES)
def test_get_sequential_output_size_non_last(output_size: int):
    assert (
        _get_sequential_output_size(nn.Sequential(nn.Linear(6, output_size), nn.ReLU()))
        == output_size
    )


def test_get_sequential_output_size_error():
    with raises(TypeError):
        _get_sequential_output_size(nn.Sequential(nn.Dropout(0.5), nn.ReLU()))


def test_get_sequential_output_size_incorrect_input():
    with raises(TypeError):
        _get_sequential_output_size(nn.Dropout(0.5))


####################################
#     Tests for has_batch_norm     #
####################################


@mark.parametrize(
    "module",
    (
        nn.BatchNorm1d(12),
        nn.BatchNorm2d(12),
        nn.BatchNorm3d(12),
        nn.SyncBatchNorm(12),
        nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6)),
    ),
)
def test_has_batch_norm_true(module: nn.Module):
    assert has_batch_norm(module)


@mark.parametrize("module", (nn.Linear(4, 6), nn.Sequential(nn.Linear(4, 6), nn.ReLU())))
def test_has_batch_norm_false(module: nn.Module):
    assert not has_batch_norm(module)


#####################################
#     Tests for get_module_name     #
#####################################


@mark.parametrize("module,name", ((nn.ReLU(), "ReLU"), (nn.Linear(4, 6), "Linear")))
def test_get_module_name(name: str, module: nn.Module):
    assert get_module_name(module) == name


####################################
#     Tests for is_batch_first     #
####################################


@mark.parametrize(
    "module",
    (
        nn.GRU(input_size=4, hidden_size=8, batch_first=True),
        nn.MultiheadAttention(4, 1, batch_first=True),
        nn.TransformerEncoderLayer(
            d_model=4,
            nhead=1,
            dim_feedforward=8,
            batch_first=True,
        ),
        nn.TransformerDecoderLayer(
            d_model=4,
            nhead=1,
            dim_feedforward=8,
            batch_first=True,
        ),
        nn.TransformerEncoder(
            num_layers=2,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=4,
                nhead=1,
                dim_feedforward=8,
                batch_first=True,
            ),
        ),
        nn.TransformerDecoder(
            num_layers=2,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=4,
                nhead=1,
                dim_feedforward=8,
                batch_first=True,
            ),
        ),
    ),
)
def test_is_batch_first_true(module: nn.Module):
    assert is_batch_first(module)


@mark.parametrize(
    "module",
    (
        nn.GRU(input_size=4, hidden_size=8, batch_first=False),
        nn.MultiheadAttention(4, 1, batch_first=False),
        nn.TransformerEncoderLayer(
            d_model=4,
            nhead=1,
            dim_feedforward=8,
            batch_first=False,
        ),
        nn.TransformerDecoderLayer(
            d_model=4,
            nhead=1,
            dim_feedforward=8,
            batch_first=False,
        ),
        nn.TransformerEncoder(
            num_layers=2,
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=4,
                nhead=1,
                dim_feedforward=8,
                batch_first=False,
            ),
        ),
        nn.TransformerDecoder(
            num_layers=2,
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=4,
                nhead=1,
                dim_feedforward=8,
                batch_first=False,
            ),
        ),
    ),
)
def test_is_batch_first_false(module: nn.Module):
    assert not is_batch_first(module)


def test_is_batch_first_incorrect_type():
    with raises(TypeError):
        is_batch_first(nn.Linear(4, 8))


#################################
#     Tests for module_mode     #
#################################


def test_module_mode_train():
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training
    with module_mode(module):
        module.train()
        assert module.training
        assert module["module1"].training
        assert module["module2"].training
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


def test_module_mode_eval():
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training
    with module_mode(module):
        module.eval()
        assert not module.training
        assert not module["module1"].training
        assert not module["module2"].training
    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


def test_module_mode_with_exception():
    module = nn.ModuleDict({"module1": nn.Linear(4, 6), "module2": nn.Linear(2, 4).eval()})
    assert module.training
    with raises(RuntimeError):
        with module_mode(module):
            module.eval()
            assert not module.training
            assert not module["module1"].training
            assert not module["module2"].training
            raise RuntimeError

    assert module.training
    assert module["module1"].training
    assert not module["module2"].training


#####################################
#     Tests for top_module_mode     #
#####################################


def test_top_module_mode_train():
    module = nn.Linear(4, 6)
    assert module.training
    with top_module_mode(module):
        module.eval()
        assert not module.training
    assert module.training


def test_top_module_mode_eval():
    module = nn.Linear(4, 6)
    module.eval()
    assert not module.training
    with top_module_mode(module):
        module.train()
        assert module.training
    assert not module.training


def test_top_module_mode_with_exception():
    module = nn.Linear(4, 6)
    assert module.training
    with raises(RuntimeError):
        with top_module_mode(module):
            module.eval()
            assert not module.training
            raise RuntimeError
    assert module.training
