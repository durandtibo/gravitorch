import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import torch
from coola import objects_are_equal
from pytest import LogCaptureFixture, mark, raises
from torch import Tensor, nn

from gravitorch import constants as ct
from gravitorch.nn import (
    find_module_state_dict,
    load_checkpoint_to_module,
    state_dicts_are_equal,
)
from gravitorch.nn.utils import (
    load_module_state_dict,
    load_state_dict_to_module,
    show_state_dict_info,
)
from gravitorch.utils import get_available_devices

LINEAR_STATE_DICT = {
    "weight": torch.ones(5, 4),
    "bias": 2 * torch.ones(5),
}

STATE_DICTS = [
    {"model": {"network": LINEAR_STATE_DICT}},
    {"list": ["weight", "bias"], "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {"set": {"weight", "bias"}, "model": {"network": LINEAR_STATE_DICT}},  # should not be detected
    {
        "tuple": ("weight", "bias"),
        "model": {"network": LINEAR_STATE_DICT},
    },  # should not be detected
    {"list": ["weight", "bias", LINEAR_STATE_DICT], "abc": None},
]


############################################
#     Tests for find_module_state_dict     #
############################################


def test_find_module_state_dict() -> None:
    module = nn.Linear(4, 5)
    state_dict = {
        "weight": torch.ones(5, 4),
        "bias": 2 * torch.ones(5),
    }
    assert objects_are_equal(
        state_dict, find_module_state_dict(state_dict, set(module.state_dict().keys()))
    )


@mark.parametrize("state_dict", STATE_DICTS)
def test_find_module_state_dict_nested(state_dict: dict) -> None:
    assert objects_are_equal(
        LINEAR_STATE_DICT, find_module_state_dict(state_dict, {"bias", "weight"})
    )


def test_find_module_state_dict_missing_key() -> None:
    assert find_module_state_dict({"weight": torch.ones(5, 4)}, {"bias", "weight"}) == {}


###############################################
#     Tests for load_checkpoint_to_module     #
###############################################


@mark.parametrize("device_weights", get_available_devices())
@mark.parametrize("device_module", get_available_devices())
def test_load_checkpoint_to_module_devices(
    tmp_path: Path, device_weights: str, device_module: str
) -> None:
    # This test verifies that it is possible to load weights from any devices to a model on any devices.
    device_weights = torch.device(device_weights)
    device_module = torch.device(device_module)
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    weights = OrderedDict(
        [
            ("weight", torch.ones(5, 4, device=device_weights)),
            ("bias", 2 * torch.ones(5, device=device_weights)),
        ]
    )
    torch.save(weights, checkpoint_path)

    module = nn.Linear(4, 5).to(device=device_module)
    out = module(torch.ones(2, 4, device=device_module))
    assert not out.equal(6 * torch.ones(2, 5, device=device_module))

    load_checkpoint_to_module(checkpoint_path, module)
    out = module(torch.ones(2, 4, device=device_module))
    assert out.equal(6 * torch.ones(2, 5, device=device_module))


@mark.parametrize("state_dict", STATE_DICTS)
def test_load_checkpoint_to_module_find_module(tmp_path: Path, state_dict: dict) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(state_dict, checkpoint_path)

    module = nn.Linear(4, 5)
    load_checkpoint_to_module(checkpoint_path, module)

    out = module(torch.ones(2, 4))
    assert out.equal(6 * torch.ones(2, 5))


def test_load_checkpoint_to_module_incorrect_path(tmp_path: Path) -> None:
    with raises(FileNotFoundError, match="No such file or directory:"):
        load_checkpoint_to_module(tmp_path.joinpath("checkpoint.pt"), nn.Linear(4, 5))


def test_load_checkpoint_to_module_incompatible_module(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(LINEAR_STATE_DICT, checkpoint_path)
    with raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_checkpoint_to_module(checkpoint_path, nn.Linear(6, 10))


def test_load_checkpoint_to_module_partial_state_dict(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(OrderedDict([("weight", torch.ones(5, 4))]), checkpoint_path)
    with raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_checkpoint_to_module(checkpoint_path, nn.Linear(4, 5))


def test_load_checkpoint_to_module_dict_strict_false_partial_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(OrderedDict([("weight", torch.ones(5, 4))]), checkpoint_path)

    module = nn.Linear(4, 5)
    load_checkpoint_to_module(checkpoint_path, module, strict=False)

    out = module(torch.ones(2, 4))
    assert out.shape == (
        2,
        5,
    )  # The bias is randomly initialized so it is not possible to know the exact value.
    assert module.weight.equal(torch.ones(5, 4))


def test_load_checkpoint_to_module_key_str(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    weights = {"model": LINEAR_STATE_DICT}
    torch.save(weights, checkpoint_path)

    module = nn.Linear(4, 5)
    load_checkpoint_to_module(checkpoint_path, module, key="model")

    out = module(torch.ones(2, 4))
    assert out.equal(6 * torch.ones(2, 5))


@mark.parametrize("key", [["model", "network"], ("model", "network")])
def test_load_checkpoint_to_module_key_sequence(tmp_path: Path, key: Sequence[str]) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    weights = {"model": {"network": LINEAR_STATE_DICT}}
    torch.save(weights, checkpoint_path)

    module = nn.Linear(4, 5)
    load_checkpoint_to_module(checkpoint_path, module, key=key)

    assert module(torch.ones(2, 4)).equal(6 * torch.ones(2, 5))


class MyNetwork(nn.Module):
    def __init__(self, checkpoint_path: Union[Path, str, None] = None) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 5)
        if checkpoint_path:
            load_checkpoint_to_module(checkpoint_path, self)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def test_load_checkpoint_to_module_my_network(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(
        {
            "fc.weight": torch.ones(5, 4),
            "fc.bias": 2 * torch.ones(5),
        },
        checkpoint_path,
    )

    # Does not load the weights
    module = MyNetwork()
    assert not module(torch.ones(2, 4)).equal(6 * torch.ones(2, 5))

    # Load the weights
    module = MyNetwork(checkpoint_path=checkpoint_path)
    assert module(torch.ones(2, 4)).equal(6 * torch.ones(2, 5))


###############################################
#     Tests for load_state_dict_to_module     #
###############################################


@mark.parametrize("state_dict", STATE_DICTS)
def test_load_state_dict_to_module_find_module(tmp_path: Path, state_dict: dict) -> None:
    module = nn.Linear(4, 5)
    load_state_dict_to_module(state_dict, module)
    out = module(torch.ones(2, 4))
    assert out.equal(6 * torch.ones(2, 5))


def test_load_state_dict_to_module_incompatible_module() -> None:
    with raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_state_dict_to_module(LINEAR_STATE_DICT, nn.Linear(6, 10))


def test_load_state_dict_to_module_partial_state_dict() -> None:
    with raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_state_dict_to_module(OrderedDict([("weight", torch.ones(5, 4))]), nn.Linear(4, 5))


def test_load_state_dict_to_module_dict_strict_false_partial_state() -> None:
    module = nn.Linear(4, 5)
    load_state_dict_to_module(OrderedDict([("weight", torch.ones(5, 4))]), module, strict=False)

    out = module(torch.ones(2, 4))
    assert out.shape == (
        2,
        5,
    )  # The bias is randomly initialized so it is not possible to know the exact value.
    assert module.weight.equal(torch.ones(5, 4))


###########################################
#     Tests for state_dicts_are_equal     #
###########################################


def test_state_dicts_are_equal_true() -> None:
    module1 = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    module2 = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6))
    module2.load_state_dict(module1.state_dict())
    assert state_dicts_are_equal(module1, module2)


def test_state_dicts_are_equal_false() -> None:
    assert not state_dicts_are_equal(
        nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)),
        nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6)),
    )


##########################################
#     Tests for show_state_dict_info     #
##########################################


def test_show_state_dict_info_empty(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_state_dict_info({})


def test_show_state_dict_info_linear(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_state_dict_info(nn.Linear(4, 6).state_dict())


def test_show_state_dict_info_no_tensor(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        show_state_dict_info({"a": 1})


###########################################
#     Tests for load_module_state_dict     #
###########################################


@mark.parametrize("device_weights", get_available_devices())
@mark.parametrize("device_module", get_available_devices())
def test_load_module_state_dict_devices(
    tmp_path: Path, device_weights: str, device_module: str
) -> None:
    # This test verifies that it is possible to load weights from any devices to a model
    # on any devices.
    device_weights = torch.device(device_weights)
    device_module = torch.device(device_module)
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    weights = {
        "modules": {
            ct.MODEL: OrderedDict(
                [
                    ("weight", torch.ones(5, 4, device=device_weights)),
                    ("bias", 2 * torch.ones(5, device=device_weights)),
                ]
            )
        }
    }
    torch.save(weights, checkpoint_path)

    module = nn.Linear(4, 5).to(device=device_module)
    assert not module(torch.ones(2, 4, device=device_module)).equal(
        6 * torch.ones(2, 5, device=device_module)
    )

    load_module_state_dict(checkpoint_path, module)
    assert module(torch.ones(2, 4, device=device_module)).equal(
        6 * torch.ones(2, 5, device=device_module)
    )


def test_load_module_state_dict_dict_strict_false_partial_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(
        {
            "modules": {
                ct.MODEL: OrderedDict(
                    [
                        ("weight", torch.ones(5, 4)),
                        ("bias", 2 * torch.ones(5)),
                    ]
                )
            }
        },
        checkpoint_path,
    )

    module = nn.Linear(4, 5)
    load_module_state_dict(checkpoint_path, module, strict=False, exclude_key_prefixes=["bias"])
    assert not module.bias.equal(torch.ones(5))
    assert module.weight.equal(torch.ones(5, 4))


def test_load_module_state_dict_dict_strict_true_partial_state(tmp_path: Path) -> None:
    checkpoint_path = tmp_path.joinpath("checkpoint.pt")
    torch.save(
        {
            "modules": {
                ct.MODEL: OrderedDict(
                    [
                        ("weight", torch.ones(5, 4)),
                        ("bias", 2 * torch.ones(5)),
                    ]
                )
            }
        },
        checkpoint_path,
    )

    module = nn.Linear(4, 5)
    with raises(RuntimeError, match=r"Error\(s\) in loading state_dict for Linear:"):
        load_module_state_dict(checkpoint_path, module, strict=True, exclude_key_prefixes=["bias"])
