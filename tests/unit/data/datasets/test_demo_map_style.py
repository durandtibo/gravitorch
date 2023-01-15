import torch
from pytest import mark, raises

from gravitorch import constants as ct
from gravitorch.data.datasets.demo_map_style import DemoMultiClassClsDataset

#############################################
#     Tests of DemoMultiClassClsDataset     #
#############################################


@mark.parametrize("num_examples", (1, 4, 8))
def test_demo_multiclass_cls_dataset_num_examples(num_examples: int):
    dataset = DemoMultiClassClsDataset(num_examples)
    assert dataset.num_examples == num_examples
    assert dataset._targets.shape[0] == num_examples
    assert dataset._features.shape[0] == num_examples


def test_demo_multiclass_cls_dataset_incorrect_num_examples():
    with raises(ValueError):
        DemoMultiClassClsDataset(num_examples=0)


@mark.parametrize("num_classes", (1, 4, 8))
def test_demo_multiclass_cls_dataset_num_classes(num_classes: int):
    dataset = DemoMultiClassClsDataset(num_examples=10, num_classes=num_classes)
    assert dataset.num_classes == num_classes
    assert torch.min(dataset._targets) >= 0
    assert torch.max(dataset._targets) < num_classes


def test_demo_multiclass_cls_dataset_incorrect_num_classes():
    with raises(ValueError):
        DemoMultiClassClsDataset(num_classes=0)


@mark.parametrize("feature_size", [1, 4, 8])
def test_demo_multiclass_cls_dataset_feature_size(feature_size: int):
    dataset = DemoMultiClassClsDataset(num_examples=10, num_classes=1, feature_size=feature_size)
    assert dataset.feature_size == feature_size
    assert dataset._features.shape[1] == feature_size


def test_demo_multiclass_cls_dataset_incorrect_feature_size():
    with raises(ValueError):
        DemoMultiClassClsDataset(num_classes=50, feature_size=32)


@mark.parametrize("noise_std", (0, 0.1, 1))
def test_demo_multiclass_cls_dataset_noise_std(noise_std: float):
    assert DemoMultiClassClsDataset(num_examples=10, noise_std=noise_std).noise_std == noise_std


def test_demo_multiclass_cls_dataset_noise_std_0():
    dataset = DemoMultiClassClsDataset(num_examples=10, noise_std=0)
    assert torch.min(dataset._features) == 0
    assert torch.max(dataset._features) == 1


def test_demo_multiclass_cls_dataset_incorrect_noise_std():
    with raises(ValueError):
        DemoMultiClassClsDataset(noise_std=-1)


def test_demo_multiclass_cls_dataset_getitem():
    example = DemoMultiClassClsDataset(num_examples=10)[0]
    assert ct.INPUT in example
    assert torch.is_tensor(example[ct.INPUT])
    assert example[ct.INPUT].shape == (64,)
    assert ct.TARGET in example
    assert torch.is_tensor(example[ct.TARGET])
    assert example[ct.TARGET].shape == ()
    assert example[ct.NAME] == "0"


@mark.parametrize("num_examples", (1, 4, 8))
def test_demo_multiclass_cls_dataset_len(num_examples: int):
    dataset = DemoMultiClassClsDataset(num_examples)
    assert len(dataset) == num_examples


def test_demo_multiclass_cls_dataset_str():
    assert str(DemoMultiClassClsDataset()).startswith("DemoMultiClassClsDataset(")
