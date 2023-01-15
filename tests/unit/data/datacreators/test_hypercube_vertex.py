import torch
from coola import objects_are_equal
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.data.datacreators import HypercubeVertexDataCreator

SIZES = (1, 2, 4)


################################################
#     Tests for HypercubeVertexDataCreator     #
################################################


def test_hypercube_vertex_data_creator_str():
    assert str(HypercubeVertexDataCreator()).startswith("HypercubeVertexDataCreator(")


@mark.parametrize("num_examples", SIZES)
def test_hypercube_vertex_data_creator_num_examples(num_examples: int):
    assert HypercubeVertexDataCreator(num_examples).num_examples == num_examples


def test_hypercube_vertex_data_creator_incorrect_num_examples():
    with raises(ValueError):
        HypercubeVertexDataCreator(num_examples=0)


@mark.parametrize("num_classes", SIZES)
def test_hypercube_vertex_data_creator_num_classes(num_classes: int):
    assert (
        HypercubeVertexDataCreator(num_examples=10, num_classes=num_classes).num_classes
        == num_classes
    )


def test_hypercube_vertex_data_creator_incorrect_num_classes():
    with raises(ValueError):
        HypercubeVertexDataCreator(num_classes=0)


@mark.parametrize("feature_size", SIZES)
def test_hypercube_vertex_data_creator_feature_size(feature_size: int):
    assert (
        HypercubeVertexDataCreator(
            num_examples=10, num_classes=1, feature_size=feature_size
        ).feature_size
        == feature_size
    )


def test_hypercube_vertex_data_creator_incorrect_feature_size():
    with raises(ValueError):
        HypercubeVertexDataCreator(num_classes=50, feature_size=32)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_hypercube_vertex_data_creator_noise_std(noise_std: float):
    assert HypercubeVertexDataCreator(num_examples=10, noise_std=noise_std).noise_std == noise_std


def test_hypercube_vertex_data_creator_incorrect_noise_std():
    with raises(ValueError):
        HypercubeVertexDataCreator(noise_std=-1)


@mark.parametrize("random_seed", (42, 35))
def test_hypercube_vertex_data_creator_random_seed(random_seed: int):
    assert (
        HypercubeVertexDataCreator(num_examples=10, random_seed=random_seed).random_seed
        == random_seed
    )


def test_hypercube_vertex_data_creator_create():
    data = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=8).create()
    assert len(data) == 2
    assert data[ct.TARGET].shape == (10,)
    assert data[ct.TARGET].dtype == torch.long
    assert data[ct.INPUT].shape == (10, 8)
    assert data[ct.INPUT].dtype == torch.float


@mark.parametrize("num_examples", SIZES)
def test_hypercube_vertex_data_creator_create_num_examples(num_examples: int):
    data = HypercubeVertexDataCreator(num_examples).create()
    assert data[ct.TARGET].shape[0] == num_examples
    assert data[ct.INPUT].shape[0] == num_examples


@mark.parametrize("num_classes", SIZES)
def test_hypercube_vertex_data_creator_create_num_classes(num_classes: int):
    targets = HypercubeVertexDataCreator(num_examples=10, num_classes=num_classes).create()[
        ct.TARGET
    ]
    assert torch.min(targets) >= 0
    assert torch.max(targets) < num_classes


@mark.parametrize("feature_size", SIZES)
def test_hypercube_vertex_data_creator_create_feature_size(feature_size: int):
    data = HypercubeVertexDataCreator(
        num_examples=10, num_classes=1, feature_size=feature_size
    ).create()
    assert data[ct.INPUT].shape[1] == feature_size


def test_hypercube_vertex_data_creator_noise_std_0():
    features = HypercubeVertexDataCreator(num_examples=10, noise_std=0).create()[ct.INPUT]
    assert torch.min(features) == 0
    assert torch.max(features) == 1


def test_hypercube_vertex_data_creator_create_same_random_seed():
    creator1 = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8, random_seed=1
    )
    creator2 = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8, random_seed=1
    )
    assert objects_are_equal(creator1.create(), creator2.create())


def test_hypercube_vertex_data_creator_create_different_random_seeds():
    creator1 = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8, random_seed=1
    )
    creator2 = HypercubeVertexDataCreator(
        num_examples=10, num_classes=5, feature_size=8, random_seed=2
    )
    assert not objects_are_equal(creator1.create(), creator2.create())


def test_hypercube_vertex_data_creator_create_repeat():
    creator = HypercubeVertexDataCreator(num_examples=10, num_classes=5, feature_size=8)
    assert not objects_are_equal(creator.create(), creator.create())
