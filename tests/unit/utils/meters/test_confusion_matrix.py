import math
from unittest.mock import Mock, patch

import torch
from coola import objects_are_allclose, objects_are_equal
from pytest import mark, raises

from gravitorch.distributed.ddp import SUM
from gravitorch.utils.meters import EmptyMeterError
from gravitorch.utils.meters.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
    check_confusion_matrix,
    check_op_compatibility_binary,
    check_op_compatibility_multiclass,
)

###########################################
#     Tests for BinaryConfusionMatrix     #
###########################################


def test_binary_confusion_matrix_repr():
    assert repr(BinaryConfusionMatrix()).startswith("BinaryConfusionMatrix(")


def test_binary_confusion_matrix_str():
    assert str(BinaryConfusionMatrix()).startswith("BinaryConfusionMatrix(")


def test_binary_confusion_matrix_init_default():
    meter = BinaryConfusionMatrix()
    assert meter.matrix.equal(torch.zeros(2, 2, dtype=torch.long))
    assert meter.num_predictions == 0


def test_binary_confusion_matrix_init():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    assert meter.matrix.equal(torch.tensor([[3, 2], [1, 4]]))
    assert meter.num_predictions == 10


def test_binary_confusion_matrix_init_incorrect_ndim():
    with raises(ValueError):
        BinaryConfusionMatrix(torch.zeros(3))


def test_binary_confusion_matrix_init_incorrect_shape():
    with raises(ValueError):
        BinaryConfusionMatrix(torch.zeros(3, 5))


def test_binary_confusion_matrix_init_incorrect_dtype():
    with raises(ValueError):
        BinaryConfusionMatrix(torch.zeros(2, 2, dtype=torch.float))


def test_binary_confusion_matrix_init_negative_value():
    with raises(ValueError):
        BinaryConfusionMatrix(torch.tensor([[0, 0], [-1, 0]]))


def test_binary_confusion_matrix_num_classes():
    assert BinaryConfusionMatrix().num_classes == 2


def test_binary_confusion_matrix_all_reduce():
    meter = BinaryConfusionMatrix(torch.ones(2, 2, dtype=torch.long))
    meter.all_reduce()
    assert meter.matrix.equal(torch.ones(2, 2, dtype=torch.long))
    assert meter.num_predictions == 4


def test_binary_confusion_matrix_all_reduce_sum_reduce():
    meter = BinaryConfusionMatrix(torch.ones(2, 2, dtype=torch.long))
    with patch("gravitorch.utils.meters.confusion_matrix.sync_reduce_") as reduce_mock:
        meter.all_reduce()
        assert objects_are_equal(
            reduce_mock.call_args.args, (torch.ones(2, 2, dtype=torch.long), SUM)
        )


def test_binary_confusion_matrix_clone():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]], dtype=torch.long))
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(meter_cloned)


def test_binary_confusion_matrix_equal_true():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    )


def test_binary_confusion_matrix_equal_false_different_values():
    assert not BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [0, 4]]))
    )


def test_binary_confusion_matrix_equal_false_different_type():
    assert not BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(42)


def test_binary_confusion_matrix_get_normalized_matrix_normalization_true():
    assert (
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        .get_normalized_matrix(normalization="true")
        .equal(torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_normalization_true_empty():
    assert (
        BinaryConfusionMatrix()
        .get_normalized_matrix(normalization="true")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_normalization_pred():
    assert (
        BinaryConfusionMatrix(torch.tensor([[3, 6], [1, 4]]))
        .get_normalized_matrix(normalization="pred")
        .equal(torch.tensor([[0.75, 0.6], [0.25, 0.4]], dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_normalization_pred_empty():
    assert (
        BinaryConfusionMatrix()
        .get_normalized_matrix(normalization="pred")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_normalization_all():
    assert (
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        .get_normalized_matrix(normalization="all")
        .equal(torch.tensor([[0.3, 0.2], [0.1, 0.4]], dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_normalization_all_empty():
    assert (
        BinaryConfusionMatrix()
        .get_normalized_matrix(normalization="all")
        .equal(torch.zeros(2, 2, dtype=torch.float))
    )


def test_binary_confusion_matrix_get_normalized_matrix_incorrect_normalization():
    with raises(ValueError):
        BinaryConfusionMatrix().get_normalized_matrix(normalization="incorrect")


def test_binary_confusion_matrix_reset():
    meter = BinaryConfusionMatrix(torch.ones(2, 2, dtype=torch.long))
    meter.reset()
    assert meter.matrix.equal(torch.zeros(2, 2, dtype=torch.long))
    assert meter.num_predictions == 0


def test_binary_confusion_matrix_sync_update_matrix():
    meter = BinaryConfusionMatrix(torch.ones(2, 2, dtype=torch.long))
    with patch(
        "gravitorch.utils.meters.confusion_matrix.sync_reduce_",
        lambda variable, op: variable.mul_(4),
    ):
        meter.all_reduce()
    assert meter.matrix.equal(torch.ones(2, 2, dtype=torch.long).mul(4))
    assert meter.num_predictions == 16


def test_binary_confusion_matrix_update():
    meter = BinaryConfusionMatrix()
    meter.update(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 0], [1, 3]], dtype=torch.long))


def test_binary_confusion_matrix_update_2():
    meter = BinaryConfusionMatrix()
    meter.update(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    )
    meter.update(
        prediction=torch.tensor([1, 1], dtype=torch.long),
        target=torch.tensor([0, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 2], [1, 3]], dtype=torch.long))


def test_binary_confusion_matrix_from_predictions():
    assert BinaryConfusionMatrix.from_predictions(
        prediction=torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([1, 1, 1, 0, 0, 1], dtype=torch.long),
    ).matrix.equal(torch.tensor([[2, 0], [1, 3]], dtype=torch.long))


# **************************
# *     Transformation     *
# **************************


def test_binary_confusion_matrix__add__():
    assert (
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        + BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    ).equal(BinaryConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))


def test_binary_confusion_matrix__iadd__():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter += BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))


def test_binary_confusion_matrix__sub__():
    assert (
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        - BinaryConfusionMatrix(torch.tensor([[1, 0], [1, 2]]))
    ).equal(BinaryConfusionMatrix(torch.tensor([[2, 2], [0, 2]])))


def test_binary_confusion_matrix_add():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).add(
        BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    )
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))
    assert meter.num_predictions == 20


def test_binary_confusion_matrix_add_():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter.add_(BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]])))
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))
    assert meter.num_predictions == 20


def test_binary_confusion_matrix_merge():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter_merged = meter.merge(
        [
            BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]])),
            BinaryConfusionMatrix(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])))
    assert meter.num_predictions == 10
    assert meter_merged.equal(BinaryConfusionMatrix(torch.tensor([[5, 2], [8, 7]])))
    assert meter_merged.num_predictions == 22


def test_binary_confusion_matrix_merge_():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter.merge_(
        [
            BinaryConfusionMatrix(torch.tensor([[1, 0], [7, 2]])),
            BinaryConfusionMatrix(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[5, 2], [8, 7]])))
    assert meter.num_predictions == 22


def test_binary_confusion_matrix_sub():
    meter = BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).sub(
        BinaryConfusionMatrix(torch.tensor([[1, 0], [1, 2]]))
    )
    assert meter.equal(BinaryConfusionMatrix(torch.tensor([[2, 2], [0, 2]])))
    assert meter.num_predictions == 6


# *******************
# *     Metrics     *
# *******************


def test_binary_confusion_matrix_false_negative():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).false_negative == 2


def test_binary_confusion_matrix_false_positive():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).false_positive == 1


def test_binary_confusion_matrix_negative():
    assert BinaryConfusionMatrix(torch.tensor([[3, 5], [1, 4]])).negative == 5


def test_binary_confusion_matrix_positive():
    assert BinaryConfusionMatrix(torch.tensor([[3, 5], [1, 4]])).positive == 8


def test_binary_confusion_matrix_predictive_negative():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).predictive_negative == 6


def test_binary_confusion_matrix_predictive_positive():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).predictive_positive == 4


def test_binary_confusion_matrix_true_negative():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).true_negative == 4


def test_binary_confusion_matrix_true_positive():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).true_positive == 3


def test_binary_confusion_matrix_accuracy():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).accuracy() == 0.7


def test_binary_confusion_matrix_accuracy_imbalanced():
    assert math.isclose(
        BinaryConfusionMatrix(torch.tensor([[30, 2], [1, 4]])).accuracy(), 0.918918918918919
    )


def test_binary_confusion_matrix_accuracy_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().accuracy()


def test_binary_confusion_matrix_balanced_accuracy():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).balanced_accuracy() == 0.7


def test_binary_confusion_matrix_balanced_accuracy_imbalanced():
    assert BinaryConfusionMatrix(torch.tensor([[30, 2], [1, 4]])).balanced_accuracy() == 0.86875


def test_binary_confusion_matrix_balanced_accuracy_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().balanced_accuracy()


def test_binary_confusion_matrix_f_beta_score_1():
    assert math.isclose(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).f_beta_score(), 0.6666666666666666
    )


def test_binary_confusion_matrix_f_beta_score_2():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).f_beta_score(beta=2) == 0.625


def test_binary_confusion_matrix_f_beta_score_0_5():
    assert math.isclose(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).f_beta_score(beta=0.5),
        0.7142857142857143,
    )


def test_binary_confusion_matrix_f_beta_score_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().f_beta_score()


def test_binary_confusion_matrix_false_negative_rate():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).false_negative_rate() == 0.4


def test_binary_confusion_matrix_false_negative_rate_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().false_negative_rate()


def test_binary_confusion_matrix_false_positive_rate():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).false_positive_rate() == 0.2


def test_binary_confusion_matrix_false_positive_rate_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().false_positive_rate()


def test_binary_confusion_matrix_jaccard_index():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).jaccard_index() == 0.5


def test_binary_confusion_matrix_jaccard_index_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().jaccard_index()


def test_binary_confusion_matrix_precision():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).precision() == 0.75


def test_binary_confusion_matrix_precision_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().precision()


def test_binary_confusion_matrix_recall():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).recall() == 0.6


def test_binary_confusion_matrix_recall_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().recall()


def test_binary_confusion_matrix_true_negative_rate():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).true_negative_rate() == 0.8


def test_binary_confusion_matrix_true_negative_rate_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().true_negative_rate()


def test_binary_confusion_matrix_true_positive_rate():
    assert BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).true_positive_rate() == 0.6


def test_binary_confusion_matrix_true_positive_rate_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().true_positive_rate()


def test_binary_confusion_matrix_compute_all_metrics():
    assert objects_are_allclose(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).compute_all_metrics(),
        {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "false_negative_rate": 0.4,
            "false_positive_rate": 0.2,
            "jaccard_index": 0.5,
            "precision": 0.75,
            "recall": 0.6,
            "true_negative_rate": 0.8,
            "true_positive_rate": 0.6,
            "f1_score": 0.6666666666666666,
        },
    )


def test_binary_confusion_matrix_compute_all_metrics_betas():
    assert objects_are_allclose(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).compute_all_metrics(betas=(1, 2)),
        {
            "accuracy": 0.7,
            "balanced_accuracy": 0.7,
            "false_negative_rate": 0.4,
            "false_positive_rate": 0.2,
            "jaccard_index": 0.5,
            "precision": 0.75,
            "recall": 0.6,
            "true_negative_rate": 0.8,
            "true_positive_rate": 0.6,
            "f1_score": 0.6666666666666666,
            "f2_score": 0.625,
        },
    )


def test_binary_confusion_matrix_compute_all_metrics_prefix_suffix():
    assert objects_are_allclose(
        BinaryConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).compute_all_metrics(
            prefix="prefix_", suffix="_suffix"
        ),
        {
            "prefix_accuracy_suffix": 0.7,
            "prefix_balanced_accuracy_suffix": 0.7,
            "prefix_false_negative_rate_suffix": 0.4,
            "prefix_false_positive_rate_suffix": 0.2,
            "prefix_jaccard_index_suffix": 0.5,
            "prefix_precision_suffix": 0.75,
            "prefix_recall_suffix": 0.6,
            "prefix_true_negative_rate_suffix": 0.8,
            "prefix_true_positive_rate_suffix": 0.6,
            "prefix_f1_score_suffix": 0.6666666666666666,
        },
    )


def test_binary_confusion_matrix_compute_all_metrics_empty():
    with raises(EmptyMeterError):
        BinaryConfusionMatrix().compute_all_metrics()


###############################################
#     Tests for MulticlassConfusionMatrix     #
###############################################


def test_multiclass_confusion_matrix_repr():
    assert repr(MulticlassConfusionMatrix.from_num_classes(num_classes=5)).startswith(
        "MulticlassConfusionMatrix("
    )


def test_multiclass_confusion_matrix_str():
    assert str(MulticlassConfusionMatrix.from_num_classes(num_classes=5)).startswith(
        "MulticlassConfusionMatrix("
    )


def test_multiclass_confusion_matrix_init_incorrect_ndim():
    with raises(ValueError):
        MulticlassConfusionMatrix(torch.zeros(3))


def test_multiclass_confusion_matrix_init_incorrect_shape():
    with raises(ValueError):
        MulticlassConfusionMatrix(torch.zeros(3, 5))


def test_multiclass_confusion_matrix_init_incorrect_dtype():
    with raises(ValueError):
        MulticlassConfusionMatrix(torch.zeros(3, 3, dtype=torch.float))


def test_multiclass_confusion_matrix_init_negative_value():
    with raises(ValueError):
        MulticlassConfusionMatrix(torch.tensor([[0, 0], [-1, 0]]))


@mark.parametrize("num_classes", (2, 5))
def test_multiclass_confusion_matrix_num_classes(num_classes: int):
    assert (
        MulticlassConfusionMatrix.from_num_classes(num_classes=num_classes).num_classes
        == num_classes
    )


def test_multiclass_confusion_matrix_auto_update_resize():
    meter = MulticlassConfusionMatrix(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.auto_update(torch.tensor([4, 2]), torch.tensor([4, 2]))
    assert meter.matrix.equal(
        torch.tensor(
            [
                [2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 2, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=torch.long,
        )
    )
    assert meter.num_predictions == 8


def test_multiclass_confusion_matrix_auto_update_no_resize():
    meter = MulticlassConfusionMatrix(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.auto_update(torch.tensor([1, 2]), torch.tensor([1, 2]))
    assert meter.matrix.equal(torch.tensor([[2, 1, 0], [0, 1, 0], [1, 1, 2]], dtype=torch.long))
    assert meter.num_predictions == 8


def test_multiclass_confusion_matrix_clone():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]], dtype=torch.long))
    meter_cloned = meter.clone()
    assert meter is not meter_cloned
    assert meter.equal(meter_cloned)


def test_multiclass_confusion_matrix_equal_true():
    assert MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(
        MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    )


def test_multiclass_confusion_matrix_equal_false_different_values():
    assert not MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(
        MulticlassConfusionMatrix(torch.tensor([[3, 2], [0, 4]]))
    )


def test_multiclass_confusion_matrix_equal_false_different_type():
    assert not MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).equal(42)


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_true():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .get_normalized_matrix(normalization="true")
        .equal(torch.tensor([[0.3, 0.2, 0.5], [0.2, 0.8, 0.0], [0.4, 0.2, 0.4]], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_true_empty():
    assert (
        MulticlassConfusionMatrix(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="true")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_pred():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 1], [4, 2, 4]], dtype=torch.long))
        .get_normalized_matrix(normalization="pred")
        .equal(
            torch.tensor(
                [[0.375, 0.25, 0.5], [0.125, 0.5, 0.1], [0.5, 0.25, 0.4]], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_pred_empty():
    assert (
        MulticlassConfusionMatrix(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="pred")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_all():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .get_normalized_matrix(normalization="all")
        .equal(
            torch.tensor(
                [[0.12, 0.08, 0.2], [0.04, 0.16, 0.0], [0.16, 0.08, 0.16]], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_normalization_all_empty():
    assert (
        MulticlassConfusionMatrix(torch.zeros(3, 3, dtype=torch.long))
        .get_normalized_matrix(normalization="all")
        .equal(torch.zeros(3, 3, dtype=torch.float))
    )


def test_multiclass_confusion_matrix_get_normalized_matrix_incorrect_normalization():
    with raises(ValueError):
        MulticlassConfusionMatrix(torch.zeros(3, 3, dtype=torch.long)).get_normalized_matrix(
            normalization="incorrect"
        )


def test_multiclass_confusion_matrix_reset():
    meter = MulticlassConfusionMatrix(torch.ones(3, 3, dtype=torch.long))
    meter.reset()
    assert meter.matrix.equal(torch.zeros(3, 3, dtype=torch.long))
    assert meter.num_predictions == 0


def test_multiclass_confusion_matrix_resize():
    meter = MulticlassConfusionMatrix(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    meter.resize(num_classes=5)
    assert meter.matrix.equal(
        torch.tensor(
            [
                [2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    )


def test_multiclass_confusion_matrix_resize_incorrect_num_classes():
    meter = MulticlassConfusionMatrix(
        torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long)
    )
    with raises(ValueError):
        meter.resize(num_classes=2)


def test_multiclass_confusion_matrix_update():
    meter = MulticlassConfusionMatrix.from_num_classes(num_classes=3)
    meter.update(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long))


def test_multiclass_confusion_matrix_update_2():
    meter = MulticlassConfusionMatrix.from_num_classes(num_classes=3)
    meter.update(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    )
    meter.update(
        prediction=torch.tensor([1, 2, 0], dtype=torch.long),
        target=torch.tensor([2, 1, 0], dtype=torch.long),
    )
    assert meter.matrix.equal(torch.tensor([[3, 1, 0], [0, 0, 1], [1, 2, 1]], dtype=torch.long))


@mark.parametrize("num_classes", (2, 5))
def test_multiclass_confusion_matrix_from_num_classes(num_classes: int):
    assert MulticlassConfusionMatrix.from_num_classes(num_classes=num_classes).matrix.equal(
        torch.zeros(num_classes, num_classes, dtype=torch.long)
    )


def test_multiclass_confusion_matrix_from_num_classes_incorrect():
    with raises(ValueError):
        MulticlassConfusionMatrix.from_num_classes(num_classes=0)


def test_multiclass_confusion_matrix_from_predictions():
    assert MulticlassConfusionMatrix.from_predictions(
        prediction=torch.tensor([0, 1, 2, 0, 0, 1], dtype=torch.long),
        target=torch.tensor([2, 2, 2, 0, 0, 0], dtype=torch.long),
    ).matrix.equal(torch.tensor([[2, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.long))


# **************************
# *     Transformation     *
# **************************


def test_multiclass_confusion_matrix__add__():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        + MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    ).equal(MulticlassConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))


def test_multiclass_confusion_matrix__iadd__():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter += MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))


def test_multiclass_confusion_matrix__sub__():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
        - MulticlassConfusionMatrix(torch.tensor([[1, 0], [1, 2]]))
    ).equal(MulticlassConfusionMatrix(torch.tensor([[2, 2], [0, 2]])))


def test_multiclass_confusion_matrix_add():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).add(
        MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]]))
    )
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))
    assert meter.num_predictions == 20


def test_multiclass_confusion_matrix_add_():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter.add_(MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]])))
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[4, 2], [8, 6]])))
    assert meter.num_predictions == 20


def test_multiclass_confusion_matrix_merge():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter_merged = meter.merge(
        [
            MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]])),
            MulticlassConfusionMatrix(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])))
    assert meter.num_predictions == 10
    assert meter_merged.equal(MulticlassConfusionMatrix(torch.tensor([[5, 2], [8, 7]])))
    assert meter_merged.num_predictions == 22


def test_multiclass_confusion_matrix_merge_():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]]))
    meter.merge_(
        [
            MulticlassConfusionMatrix(torch.tensor([[1, 0], [7, 2]])),
            MulticlassConfusionMatrix(torch.tensor([[1, 0], [0, 1]])),
        ]
    )
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[5, 2], [8, 7]])))
    assert meter.num_predictions == 22


def test_multiclass_confusion_matrix_sub():
    meter = MulticlassConfusionMatrix(torch.tensor([[3, 2], [1, 4]])).sub(
        MulticlassConfusionMatrix(torch.tensor([[1, 0], [1, 2]]))
    )
    assert meter.equal(MulticlassConfusionMatrix(torch.tensor([[2, 2], [0, 2]])))
    assert meter.num_predictions == 6


# *******************
# *     Metrics     *
# *******************


def test_multiclass_confusion_matrix_false_negative():
    assert MulticlassConfusionMatrix(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).false_negative.equal(torch.tensor([7, 1, 6], dtype=torch.long))


def test_multiclass_confusion_matrix_false_positive():
    assert MulticlassConfusionMatrix(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).false_positive.equal(torch.tensor([5, 4, 5], dtype=torch.long))


def test_multiclass_confusion_matrix_support():
    assert MulticlassConfusionMatrix(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).support.equal(torch.tensor([10, 5, 10], dtype=torch.long))


def test_multiclass_confusion_matrix_true_positive():
    assert MulticlassConfusionMatrix(
        torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
    ).true_positive.equal(torch.tensor([3, 4, 4], dtype=torch.long))


def test_multiclass_confusion_matrix_accuracy():
    assert (
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).accuracy()
        == 0.44
    )


def test_multiclass_confusion_matrix_accuracy_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).accuracy()


def test_multiclass_confusion_matrix_balanced_accuracy():
    assert (
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).balanced_accuracy()
        == 0.5
    )


def test_multiclass_confusion_matrix_balanced_accuracy_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).balanced_accuracy()


def test_multiclass_confusion_matrix_f_beta_score_1():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .f_beta_score()
        .allclose(
            torch.tensor(
                [0.3333333333333333, 0.6153846153846154, 0.42105263157894735], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_2():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .f_beta_score(beta=2)
        .allclose(
            torch.tensor([0.3125, 0.7142857142857143, 0.40816326530612246], dtype=torch.float)
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_0_5():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .f_beta_score(beta=0.5)
        .allclose(
            torch.tensor(
                [0.35714285714285715, 0.5405405405405406, 0.43478260869565216], dtype=torch.float
            )
        )
    )


def test_multiclass_confusion_matrix_f_beta_score_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).f_beta_score()


def test_multiclass_confusion_matrix_macro_f_beta_score_1():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(),
        0.4565901756286621,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_2():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(beta=2),
        0.4783163368701935,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_0_5():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_f_beta_score(beta=0.5),
        0.44415533542633057,
    )


def test_multiclass_confusion_matrix_macro_f_beta_score_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).macro_f_beta_score()


def test_multiclass_confusion_matrix_micro_f_beta_score_1():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_2():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(beta=2),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_0_5():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_f_beta_score(beta=0.5),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_f_beta_score_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).micro_f_beta_score()


def test_multiclass_confusion_matrix_weighted_f_beta_score_1():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(),
        0.42483131408691405,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_2():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(beta=2),
        0.4311224365234375,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_0_5():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_f_beta_score(beta=0.5),
        0.4248783111572266,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_f_beta_score_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).weighted_f_beta_score()


def test_multiclass_confusion_matrix_precision():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 1], [4, 2, 4]], dtype=torch.long))
        .precision()
        .equal(torch.tensor([0.375, 0.5, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_precision_zero():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 0, 5], [3, 0, 1], [4, 0, 4]], dtype=torch.long))
        .precision()
        .equal(torch.tensor([0.3, 0.0, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_precision_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).precision()


def test_multiclass_confusion_matrix_macro_precision():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_precision(),
        0.43981480598449707,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_macro_precision_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).macro_precision()


def test_multiclass_confusion_matrix_micro_precision():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_precision(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_precision_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).micro_precision()


def test_multiclass_confusion_matrix_weighted_precision():
    assert (
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_precision()
        == 0.42777778625488283
    )


def test_multiclass_confusion_matrix_weighted_precision_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).weighted_precision()


def test_multiclass_confusion_matrix_recall():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long))
        .recall()
        .equal(torch.tensor([0.3, 0.8, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_recall_zero():
    assert (
        MulticlassConfusionMatrix(torch.tensor([[3, 2, 5], [0, 0, 0], [4, 2, 4]], dtype=torch.long))
        .recall()
        .equal(torch.tensor([0.3, 0.0, 0.4], dtype=torch.float))
    )


def test_multiclass_confusion_matrix_recall_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).recall()


def test_multiclass_confusion_matrix_macro_recall():
    assert (
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).macro_recall()
        == 0.5
    )


def test_multiclass_confusion_matrix_macro_recall_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).macro_recall()


def test_multiclass_confusion_matrix_micro_recall():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).micro_recall(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_micro_recall_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).micro_recall()


def test_multiclass_confusion_matrix_weighted_recall():
    assert math.isclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).weighted_recall(),
        0.44,
        abs_tol=1e-6,
    )


def test_multiclass_confusion_matrix_weighted_recall_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).weighted_recall()


def test_multiclass_confusion_matrix_compute_per_class_metrics():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(),
        {
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_betas():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(betas=(1, 2)),
        {
            "f1_score": torch.tensor([0.3333333432674408, 0.6153846383094788, 0.42105263471603394]),
            "f2_score": torch.tensor([0.3125, 0.7142857313156128, 0.40816327929496765]),
            "precision": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "recall": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_prefix_suffix():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_per_class_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_f1_score_suffix": torch.tensor(
                [0.3333333432674408, 0.6153846383094788, 0.42105263471603394]
            ),
            "prefix_precision_suffix": torch.tensor([0.375, 0.5, 0.4444444477558136]),
            "prefix_recall_suffix": torch.tensor([0.3, 0.8, 0.4]),
        },
    )


def test_multiclass_confusion_matrix_compute_per_class_metrics_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).compute_per_class_metrics()


def test_multiclass_confusion_matrix_compute_macro_metrics():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(),
        {
            "macro_f1_score": 0.4565901756286621,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_betas():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(betas=(1, 2)),
        {
            "macro_f1_score": 0.4565901756286621,
            "macro_f2_score": 0.4783163368701935,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_prefix_suffix():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_macro_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_macro_f1_score_suffix": 0.4565901756286621,
            "prefix_macro_precision_suffix": 0.43981480598449707,
            "prefix_macro_recall_suffix": 0.5,
        },
    )


def test_multiclass_confusion_matrix_compute_macro_metrics_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).compute_macro_metrics()


def test_multiclass_confusion_matrix_compute_micro_metrics():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(),
        {
            "micro_f1_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_betas():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(betas=(1, 2)),
        {
            "micro_f1_score": 0.44,
            "micro_f2_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_prefix_suffix():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_micro_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_micro_f1_score_suffix": 0.44,
            "prefix_micro_precision_suffix": 0.44,
            "prefix_micro_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_micro_metrics_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).compute_micro_metrics()


def test_multiclass_confusion_matrix_compute_weighted_metrics():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(),
        {
            "weighted_f1_score": 0.42483131408691405,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_betas():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(betas=(1, 2)),
        {
            "weighted_f1_score": 0.42483131408691405,
            "weighted_f2_score": 0.4311224365234375,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_prefix_suffix():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_weighted_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_weighted_f1_score_suffix": 0.42483131408691405,
            "prefix_weighted_precision_suffix": 0.42777778625488283,
            "prefix_weighted_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_weighted_metrics_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).compute_weighted_metrics()


def test_multiclass_confusion_matrix_compute_scalar_metrics():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_betas():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(betas=(1, 2)),
        {
            "accuracy": 0.44,
            "balanced_accuracy": 0.5,
            "macro_f1_score": 0.4565901756286621,
            "macro_f2_score": 0.4783163368701935,
            "macro_precision": 0.43981480598449707,
            "macro_recall": 0.5,
            "micro_f1_score": 0.44,
            "micro_f2_score": 0.44,
            "micro_precision": 0.44,
            "micro_recall": 0.44,
            "weighted_f1_score": 0.42483131408691405,
            "weighted_f2_score": 0.4311224365234375,
            "weighted_precision": 0.42777778625488283,
            "weighted_recall": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_prefix_suffix():
    assert objects_are_allclose(
        MulticlassConfusionMatrix(
            torch.tensor([[3, 2, 5], [1, 4, 0], [4, 2, 4]], dtype=torch.long)
        ).compute_scalar_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 0.44,
            "prefix_balanced_accuracy_suffix": 0.5,
            "prefix_macro_f1_score_suffix": 0.4565901756286621,
            "prefix_macro_precision_suffix": 0.43981480598449707,
            "prefix_macro_recall_suffix": 0.5,
            "prefix_micro_f1_score_suffix": 0.44,
            "prefix_micro_precision_suffix": 0.44,
            "prefix_micro_recall_suffix": 0.44,
            "prefix_weighted_f1_score_suffix": 0.42483131408691405,
            "prefix_weighted_precision_suffix": 0.42777778625488283,
            "prefix_weighted_recall_suffix": 0.44,
        },
    )


def test_multiclass_confusion_matrix_compute_scalar_metrics_empty():
    with raises(EmptyMeterError):
        MulticlassConfusionMatrix.from_num_classes(3).compute_scalar_metrics()


############################################
#     Tests for check_confusion_matrix     #
############################################


def test_check_confusion_matrix_incorrect_ndim():
    with raises(ValueError):
        check_confusion_matrix(torch.zeros(3))


def test_check_confusion_matrix_incorrect_shape():
    with raises(ValueError):
        check_confusion_matrix(torch.zeros(3, 5))


def test_check_confusion_matrix_incorrect_dtype():
    with raises(ValueError):
        check_confusion_matrix(torch.zeros(2, 2, dtype=torch.float))


def test_check_confusion_matrix_negative_value():
    with raises(ValueError):
        check_confusion_matrix(torch.tensor([[0, 0], [-1, 0]]))


###################################################
#     Tests for check_op_compatibility_binary     #
###################################################


def test_check_op_compatibility_binary_correct():
    check_op_compatibility_binary(BinaryConfusionMatrix(), BinaryConfusionMatrix(), "op")
    # will fail if an exception is raised


def test_check_op_compatibility_binary_incorrect_type():
    with raises(TypeError):
        check_op_compatibility_binary(BinaryConfusionMatrix(), Mock(), "op")


#######################################################
#     Tests for check_op_compatibility_multiclass     #
#######################################################


def test_check_op_compatibility_multiclass_correct():
    check_op_compatibility_multiclass(
        MulticlassConfusionMatrix.from_num_classes(3),
        MulticlassConfusionMatrix.from_num_classes(3),
        "op",
    )
    # will fail if an exception is raised


def test_check_op_compatibility_multiclass_incorrect_type():
    with raises(TypeError):
        check_op_compatibility_multiclass(
            MulticlassConfusionMatrix.from_num_classes(3),
            Mock(),
            "op",
        )


def test_check_op_compatibility_multiclass_incorrect_shape():
    with raises(ValueError):
        check_op_compatibility_multiclass(
            MulticlassConfusionMatrix.from_num_classes(3),
            MulticlassConfusionMatrix.from_num_classes(4),
            "op",
        )
