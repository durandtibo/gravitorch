import torch
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.data.datasets import DummyMultiClassDataset

#############################################
#     Tests of DemoMultiClassClsDataset     #
#############################################


@mark.parametrize("num_examples", (1, 4, 8))
def test_dummy_multiclass_dataset_num_examples(num_examples: int) -> None:
    dataset = DummyMultiClassDataset(num_examples)
    assert dataset.num_examples == num_examples
    assert dataset._targets.shape[0] == num_examples
    assert dataset._features.shape[0] == num_examples


def test_dummy_multiclass_dataset_incorrect_num_examples() -> None:
    with raises(ValueError, match="The number of examples (.*) has to be greater than 0"):
        DummyMultiClassDataset(num_examples=0)


@mark.parametrize("num_classes", (1, 4, 8))
def test_dummy_multiclass_dataset_num_classes(num_classes: int) -> None:
    dataset = DummyMultiClassDataset(num_examples=10, num_classes=num_classes)
    assert dataset.num_classes == num_classes
    assert torch.min(dataset._targets) >= 0
    assert torch.max(dataset._targets) < num_classes


def test_dummy_multiclass_dataset_incorrect_num_classes() -> None:
    with raises(ValueError, match="The number of classes (.*) has to be greater than 0"):
        DummyMultiClassDataset(num_classes=0)


@mark.parametrize("feature_size", [1, 4, 8])
def test_dummy_multiclass_dataset_feature_size(feature_size: int) -> None:
    dataset = DummyMultiClassDataset(num_examples=10, num_classes=1, feature_size=feature_size)
    assert dataset.feature_size == feature_size
    assert dataset._features.shape[1] == feature_size


def test_dummy_multiclass_dataset_incorrect_feature_size() -> None:
    with raises(ValueError, match="The feature dimension (.*) has to be greater or equal"):
        DummyMultiClassDataset(num_classes=50, feature_size=32)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_dummy_multiclass_dataset_noise_std(noise_std: float) -> None:
    assert DummyMultiClassDataset(num_examples=10, noise_std=noise_std).noise_std == noise_std


def test_dummy_multiclass_dataset_noise_std_0() -> None:
    dataset = DummyMultiClassDataset(num_examples=10, noise_std=0)
    assert torch.min(dataset._features) == 0
    assert torch.max(dataset._features) == 1


def test_dummy_multiclass_dataset_incorrect_noise_std() -> None:
    with raises(ValueError, match="The standard deviation of the Gaussian noise (.*)"):
        DummyMultiClassDataset(noise_std=-1)


def test_dummy_multiclass_dataset_getitem() -> None:
    example = DummyMultiClassDataset(num_examples=10)[0]
    assert ct.INPUT in example
    assert torch.is_tensor(example[ct.INPUT])
    assert example[ct.INPUT].shape == (64,)
    assert ct.TARGET in example
    assert torch.is_tensor(example[ct.TARGET])
    assert example[ct.TARGET].shape == ()
    assert example[ct.NAME] == "0"


@mark.parametrize("num_examples", (1, 4, 8))
def test_dummy_multiclass_dataset_len(num_examples: int) -> None:
    assert len(DummyMultiClassDataset(num_examples)) == num_examples


def test_dummy_multiclass_dataset_str() -> None:
    assert str(DummyMultiClassDataset()).startswith("DummyMultiClassDataset(")
