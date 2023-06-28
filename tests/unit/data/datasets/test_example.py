import logging
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from coola import EqualityTester
from pytest import LogCaptureFixture, mark

from gravitorch.data.datasets import ExampleDataset
from gravitorch.data.datasets.example import ExampleDatasetEqualityOperator
from gravitorch.utils.io import save_json, save_pickle, save_pytorch

####################################
#     Tests for ExampleDataset     #
####################################


def test_example_dataset_str() -> None:
    assert str(ExampleDataset(())).startswith("ExampleDataset")


def test_example_dataset_examples_list_to_tuple() -> None:
    assert ExampleDataset([1, 2])._examples == (1, 2)


@mark.parametrize("examples,length", (((), 0), ((1,), 1), ((1, 2), 2)))
def test_example_dataset_len(examples: Sequence, length: int) -> None:
    assert len(ExampleDataset(examples)) == length


def test_example_dataset_getitem() -> None:
    dataset = ExampleDataset((1, 2))
    assert dataset[0] == 1
    assert dataset[1] == 2


def test_example_dataset_equal_true() -> None:
    assert ExampleDataset([1, 2]).equal(ExampleDataset([1, 2]))


def test_example_dataset_equal_true_same() -> None:
    dataset = ExampleDataset([1, 2])
    assert dataset.equal(dataset)


def test_example_dataset_equal_false_different_examples() -> None:
    assert not ExampleDataset([1, 2]).equal(ExampleDataset([2, 1]))


def test_example_dataset_equal_false_different_type() -> None:
    assert not ExampleDataset([1, 2]).equal([1, 2])


def test_example_dataset_from_json_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_json((1, 2), path)
    assert ExampleDataset.from_json_file(path).equal(ExampleDataset((1, 2)))


def test_example_dataset_from_pickle_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle((1, 2), path)
    assert ExampleDataset.from_pickle_file(path).equal(ExampleDataset((1, 2)))


def test_example_dataset_from_pytorch_file(tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pt")
    save_pytorch((1, 2), path)
    assert ExampleDataset.from_pytorch_file(path).equal(ExampleDataset((1, 2)))


####################################
#     Tests for ExampleDataset     #
####################################


def test_example_dataset_equality_operator_str() -> None:
    assert str(ExampleDatasetEqualityOperator()).startswith("ExampleDatasetEqualityOperator")


def test_batch_equality_operator__eq__true() -> None:
    assert ExampleDatasetEqualityOperator() == ExampleDatasetEqualityOperator()


def test_batch_equality_operator__eq__false() -> None:
    assert ExampleDatasetEqualityOperator() != 123


def test_batch_equality_operator_clone() -> None:
    op = ExampleDatasetEqualityOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_batch_equality_operator_equal_true() -> None:
    assert ExampleDatasetEqualityOperator().equal(
        EqualityTester(), ExampleDataset((1, 2, 3)), ExampleDataset((1, 2, 3))
    )


def test_batch_equality_operator_equal_true_same_object() -> None:
    dataset = ExampleDataset((1, 2, 3))
    assert ExampleDatasetEqualityOperator().equal(EqualityTester(), dataset, dataset)


def test_batch_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with patch("gravitorch.data.datasets.example.log_box_dataset_class"):
        with caplog.at_level(logging.INFO):
            assert ExampleDatasetEqualityOperator().equal(
                tester=EqualityTester(),
                object1=ExampleDataset((1, 2, 3)),
                object2=ExampleDataset((1, 2, 3)),
                show_difference=True,
            )
            assert not caplog.messages


def test_batch_equality_operator_equal_false_different_value() -> None:
    assert not ExampleDatasetEqualityOperator().equal(
        EqualityTester(), ExampleDataset((1, 2, 3)), ExampleDataset((1, 2, 4))
    )


def test_batch_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ExampleDatasetEqualityOperator().equal(
            tester=EqualityTester(),
            object1=ExampleDataset((1, 2, 3)),
            object2=ExampleDataset((1, 2, 4)),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("`ExampleDataset` objects are different")


def test_batch_equality_operator_equal_false_different_type() -> None:
    assert not ExampleDatasetEqualityOperator().equal(
        EqualityTester(), ExampleDataset((1, 2, 3)), 42
    )


def test_batch_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ExampleDatasetEqualityOperator().equal(
            tester=EqualityTester(),
            object1=ExampleDataset((1, 2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("object2 is not a `ExampleDataset` object")
