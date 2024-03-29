from __future__ import annotations

import pickle

import torch
from coola import objects_are_equal
from pytest import mark, raises
from torch import Tensor

from gravitorch import constants as ct
from gravitorch.data.datacreators import HypercubeVertexDataCreator
from gravitorch.data.datacreators.hypercube import create_hypercube_vertex
from gravitorch.utils.seed import get_torch_generator

SIZES = (1, 2, 4)


################################################
#     Tests for HypercubeVertexDataCreator     #
################################################


def test_hypercube_vertex_data_creator_str() -> None:
    assert str(HypercubeVertexDataCreator()).startswith("HypercubeVertexDataCreator(")


@mark.parametrize("num_examples", SIZES)
def test_hypercube_vertex_data_creator_num_examples(num_examples: int) -> None:
    assert HypercubeVertexDataCreator(num_examples).num_examples == num_examples


@mark.parametrize("num_examples", (0, -1))
def test_hypercube_vertex_data_creator_incorrect_num_examples(num_examples: int) -> None:
    with raises(ValueError, match="The number of examples .* has to be greater than 0"):
        HypercubeVertexDataCreator(num_examples=num_examples)


@mark.parametrize("num_classes", SIZES)
def test_hypercube_vertex_data_creator_num_classes(num_classes: int) -> None:
    assert (
        HypercubeVertexDataCreator(num_examples=10, num_classes=num_classes).num_classes
        == num_classes
    )


@mark.parametrize("num_classes", (0, -1))
def test_hypercube_vertex_data_creator_incorrect_num_classes(num_classes: int) -> None:
    with raises(ValueError, match="he number of classes .* has to be greater than 0"):
        HypercubeVertexDataCreator(num_classes=num_classes)


@mark.parametrize("feature_size", SIZES)
def test_hypercube_vertex_data_creator_feature_size(feature_size: int) -> None:
    assert (
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=1, feature_size=feature_size
        ).feature_size
        == feature_size
    )


def test_hypercube_vertex_data_creator_incorrect_feature_size() -> None:
    with raises(
        ValueError,
        match="The feature dimension .* has to be greater or equal to the number of classes .*",
    ):
        HypercubeVertexDataCreator(num_classes=50, feature_size=32)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_hypercube_vertex_data_creator_noise_std(noise_std: float) -> None:
    assert HypercubeVertexDataCreator(num_examples=10, noise_std=noise_std).noise_std == noise_std


def test_hypercube_vertex_data_creator_incorrect_noise_std() -> None:
    with raises(
        ValueError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        HypercubeVertexDataCreator(noise_std=-1)


@mark.parametrize("random_seed", (42, 35))
def test_hypercube_vertex_data_creator_random_seed(random_seed: int) -> None:
    assert (
        HypercubeVertexDataCreator(num_examples=10, random_seed=random_seed).random_seed
        == random_seed
    )


def test_hypercube_vertex_data_creator_create() -> None:
    data = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=8).create()
    assert len(data) == 2
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.long
    assert data[ct.INPUT].shape == (10, 8)
    assert data[ct.INPUT].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_hypercube_vertex_data_creator_create_num_examples(num_examples: int) -> None:
    data = HypercubeVertexDataCreator(num_examples).create()
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.INPUT].shape[0] == num_examples


@mark.parametrize("num_classes", SIZES)
def test_hypercube_vertex_data_creator_create_num_classes(num_classes: int) -> None:
    targets = HypercubeVertexDataCreator(num_examples=10, num_classes=num_classes).create()[
        ct.TARGET
    ]
    assert torch.min(targets) >= 0
    assert torch.max(targets) < num_classes


@mark.parametrize("feature_size", SIZES)
def test_hypercube_vertex_data_creator_create_feature_size(feature_size: int) -> None:
    data = HypercubeVertexDataCreator(
        num_examples=10, num_classes=1, feature_size=feature_size
    ).create()
    assert data[ct.INPUT].shape[1] == feature_size


def test_hypercube_vertex_data_creator_noise_std_0() -> None:
    features = HypercubeVertexDataCreator(num_examples=10, noise_std=0).create()[ct.INPUT]
    assert torch.min(features) == 0
    assert torch.max(features) == 1


def test_hypercube_vertex_data_creator_create_same_random_seed() -> None:
    assert objects_are_equal(
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=5, feature_size=8, random_seed=1
        ).create(),
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=5, feature_size=8, random_seed=1
        ).create(),
    )


def test_hypercube_vertex_data_creator_create_different_random_seeds() -> None:
    assert not objects_are_equal(
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=5, feature_size=8, random_seed=1
        ).create(),
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=5, feature_size=8, random_seed=2
        ).create(),
    )


def test_hypercube_vertex_data_creator_create_log_info_false() -> None:
    data = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8, log_info=False
    ).create()
    assert len(data) == 2
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.long
    assert data[ct.INPUT].shape == (10, 8)
    assert data[ct.INPUT].dtype == torch.float


def test_hypercube_vertex_data_creator_create_repeat() -> None:
    creator = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=8)
    assert not objects_are_equal(creator.create(), creator.create())


def test_hypercube_vertex_data_creator_getstate() -> None:
    state = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8
    ).__getstate__()
    assert len(state) == 6
    assert state["_num_examples"] == 10
    assert state["_num_classes"] == 5
    assert state["_feature_size"] == 8
    assert state["_noise_std"] == 0.2
    assert state["_log_info"]
    assert isinstance(state["_generator"], Tensor)


def test_hypercube_vertex_data_creator_setstate() -> None:
    state = torch.Generator().get_state()
    creator = HypercubeVertexDataCreator(
        num_examples=1000,
        num_classes=50,
        feature_size=64,
        noise_std=0.2,
    )
    creator.__setstate__(
        {
            "_num_examples": 10,
            "_num_classes": 5,
            "_feature_size": 8,
            "_noise_std": 0.5,
            "_log_info": False,
            "_generator": state,
        }
    )
    assert creator.num_classes == 5
    assert creator.num_examples == 10
    assert creator.noise_std == 0.5
    assert creator.feature_size == 8
    assert not creator._log_info
    assert creator._generator.get_state().equal(state)


def test_hypercube_vertex_data_creator_state_repeat() -> None:
    creator = HypercubeVertexDataCreator(
        num_examples=20,
        num_classes=5,
        feature_size=8,
        noise_std=0.2,
    )
    state = creator.__getstate__()
    data1 = creator.create()
    data2 = creator.create()
    creator.__setstate__(state)
    data3 = creator.create()
    assert not objects_are_equal(data1, data2)
    assert objects_are_equal(data1, data3)


def test_hypercube_vertex_data_creator_pickle_dumps() -> None:
    # Verify that the object can be serialized
    pickle.dumps(HypercubeVertexDataCreator())


#############################################
#     Tests for create_hypercube_vertex     #
#############################################


@mark.parametrize("num_examples", (0, -1))
def test_create_hypercube_vertex_incorrect_num_examples(num_examples: int) -> None:
    with raises(RuntimeError, match="The number of examples .* has to be greater than 0"):
        create_hypercube_vertex(num_examples=num_examples)


@mark.parametrize("num_classes", (0, -1))
def test_create_hypercube_vertex_incorrect_num_classes(num_classes: int) -> None:
    with raises(RuntimeError, match="he number of classes .* has to be greater than 0"):
        create_hypercube_vertex(num_classes=num_classes)


def test_create_hypercube_vertex_incorrect_feature_size() -> None:
    with raises(
        RuntimeError,
        match="The feature dimension .* has to be greater or equal to the number of classes .*",
    ):
        create_hypercube_vertex(num_classes=50, feature_size=32)


def test_create_hypercube_vertex_incorrect_noise_std() -> None:
    with raises(
        RuntimeError,
        match="The standard deviation of the Gaussian noise .* has to be greater or equal than 0",
    ):
        create_hypercube_vertex(noise_std=-1)


def test_create_hypercube_vertex_create() -> None:
    data = create_hypercube_vertex(num_examples=10, num_classes=5, feature_size=8)
    assert len(data) == 2
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.long
    assert data[ct.INPUT].shape == (10, 8)
    assert data[ct.INPUT].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_create_hypercube_vertex_create_num_examples(num_examples: int) -> None:
    data = create_hypercube_vertex(num_examples)
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.INPUT].shape[0] == num_examples


@mark.parametrize("num_classes", SIZES)
def test_create_hypercube_vertex_create_num_classes(num_classes: int) -> None:
    targets = create_hypercube_vertex(num_examples=10, num_classes=num_classes)[ct.TARGET]
    assert torch.min(targets) >= 0
    assert torch.max(targets) < num_classes


@mark.parametrize("feature_size", SIZES)
def test_create_hypercube_vertex_create_feature_size(feature_size: int) -> None:
    data = create_hypercube_vertex(num_examples=10, num_classes=1, feature_size=feature_size)
    assert data[ct.INPUT].shape[1] == feature_size


def test_create_hypercube_vertex_noise_std_0() -> None:
    features = create_hypercube_vertex(num_examples=10, noise_std=0)[ct.INPUT]
    assert torch.min(features) == 0
    assert torch.max(features) == 1


def test_create_hypercube_vertex_create_same_random_seed() -> None:
    assert objects_are_equal(
        create_hypercube_vertex(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
        create_hypercube_vertex(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
    )


def test_create_hypercube_vertex_create_different_random_seeds() -> None:
    assert not objects_are_equal(
        create_hypercube_vertex(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(1)
        ),
        create_hypercube_vertex(
            num_examples=10, num_classes=5, feature_size=8, generator=get_torch_generator(2)
        ),
    )
