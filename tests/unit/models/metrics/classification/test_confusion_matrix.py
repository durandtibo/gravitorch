import torch
from pytest import fixture, mark, raises

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    BinaryConfusionMatrix,
    CategoricalConfusionMatrix,
    EmptyMetricError,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.history import MaxScalarHistory, MinScalarHistory

MODES = (ct.TRAIN, ct.EVAL)
NAMES = ("name1", "name2")
SIZES = (1, 2)
DTYPES = (torch.long, torch.float)


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


###########################################
#     Tests for BinaryConfusionMatrix     #
###########################################


@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_str(mode: str) -> None:
    assert str(BinaryConfusionMatrix(mode)).startswith("BinaryConfusionMatrix(")


@mark.parametrize("name", NAMES)
def test_binary_confusion_matrix_attach_train(name: str, engine: BaseEngine):
    metric = BinaryConfusionMatrix(ct.TRAIN, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_balanced_accuracy"), MaxScalarHistory)
    assert isinstance(
        engine.get_history(f"{ct.TRAIN}/{name}_false_negative_rate"), MinScalarHistory
    )
    assert isinstance(
        engine.get_history(f"{ct.TRAIN}/{name}_false_positive_rate"), MinScalarHistory
    )
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_jaccard_index"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_true_negative_rate"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_true_positive_rate"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_binary_confusion_matrix_attach_eval(name: str, engine: BaseEngine):
    metric = BinaryConfusionMatrix(ct.EVAL, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_balanced_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_false_negative_rate"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_false_positive_rate"), MinScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_jaccard_index"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_true_negative_rate"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_true_positive_rate"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_incorrect(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([1, 0, 1, 0], device=device))
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 0.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 0.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_jaccard_index": 0.0,
        f"{mode}/bin_conf_mat_precision": 0.0,
        f"{mode}/bin_conf_mat_recall": 0.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_f1_score": 0.0,
        f"{mode}/bin_conf_mat_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_betas(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode, betas=(0.5, 1, 2)).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f0.5_score": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_f2_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 4,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device),
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device),
    )
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_binary_confusion_matrix_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
):
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device, dtype=dtype_prediction),
        torch.tensor([[0, 1], [0, 1], [1, 0]], device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    metric(torch.tensor([1, 0], device=device), torch.tensor([1, 0], device=device))
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_forward_multiple_batches_with_reset(
    device: str, mode: str
) -> None:
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    metric.reset()
    metric(torch.tensor([1, 0], device=device), torch.tensor([1, 0], device=device))
    assert metric.value() == {
        f"{mode}/bin_conf_mat_accuracy": 1.0,
        f"{mode}/bin_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/bin_conf_mat_false_negative_rate": 0.0,
        f"{mode}/bin_conf_mat_false_positive_rate": 0.0,
        f"{mode}/bin_conf_mat_jaccard_index": 1.0,
        f"{mode}/bin_conf_mat_precision": 1.0,
        f"{mode}/bin_conf_mat_recall": 1.0,
        f"{mode}/bin_conf_mat_true_negative_rate": 1.0,
        f"{mode}/bin_conf_mat_true_positive_rate": 1.0,
        f"{mode}/bin_conf_mat_f1_score": 1.0,
        f"{mode}/bin_conf_mat_num_predictions": 2,
    }


@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_value_empty(mode):
    with raises(EmptyMetricError):
        BinaryConfusionMatrix(mode).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_binary_confusion_matrix_value_log_engine(device: str, mode: str, engine: BaseEngine):
    device = torch.device(device)
    metric = BinaryConfusionMatrix(mode).to(device=device)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    metric.value(engine)
    assert engine.get_history(f"{mode}/bin_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_false_negative_rate").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/bin_conf_mat_false_positive_rate").get_last_value() == 0.0
    assert engine.get_history(f"{mode}/bin_conf_mat_jaccard_index").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_precision").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_recall").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_true_negative_rate").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_true_positive_rate").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/bin_conf_mat_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_binary_confusion_matrix_events_train(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = BinaryConfusionMatrix(ct.TRAIN).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert (
        engine.get_history(f"{ct.TRAIN}/bin_conf_mat_false_negative_rate").get_last_value() == 0.0
    )
    assert (
        engine.get_history(f"{ct.TRAIN}/bin_conf_mat_false_positive_rate").get_last_value() == 0.0
    )
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_jaccard_index").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_true_negative_rate").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_true_positive_rate").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/bin_conf_mat_num_predictions").get_last_value() == 4


@mark.parametrize("device", get_available_devices())
def test_binary_confusion_matrix_events_eval(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = BinaryConfusionMatrix(ct.EVAL).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(torch.tensor([0, 1, 0, 1], device=device), torch.tensor([0, 1, 0, 1], device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_false_negative_rate").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_false_positive_rate").get_last_value() == 0.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_jaccard_index").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_true_negative_rate").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_true_positive_rate").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/bin_conf_mat_num_predictions").get_last_value() == 4


def test_binary_confusion_matrix_reset() -> None:
    metric = BinaryConfusionMatrix(ct.EVAL)
    metric(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    assert metric._confusion_matrix.num_predictions == 4
    metric.reset()
    assert metric._confusion_matrix.num_predictions == 0


################################################
#     Tests for CategoricalConfusionMatrix     #
################################################


@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_str(mode: str) -> None:
    assert str(CategoricalConfusionMatrix(mode, num_classes=3)).startswith(
        "CategoricalConfusionMatrix("
    )


@mark.parametrize("name", NAMES)
def test_categorical_confusion_matrix_attach_train(name: str, engine: BaseEngine):
    metric = CategoricalConfusionMatrix(ct.TRAIN, num_classes=3, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_balanced_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_macro_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_macro_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_macro_f1_score"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_micro_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_micro_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_micro_f1_score"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_weighted_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_weighted_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/{name}_weighted_f1_score"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


@mark.parametrize("name", NAMES)
def test_categorical_confusion_matrix_attach_eval(name: str, engine: BaseEngine):
    metric = CategoricalConfusionMatrix(ct.EVAL, num_classes=3, name=name)
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_balanced_accuracy"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_macro_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_macro_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_macro_f1_score"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_micro_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_micro_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_micro_f1_score"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_weighted_precision"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_weighted_recall"), MaxScalarHistory)
    assert isinstance(engine.get_history(f"{ct.EVAL}/{name}_weighted_f1_score"), MaxScalarHistory)
    assert engine.has_event_handler(
        VanillaEventHandler(metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        VanillaEventHandler(metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_correct(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 5,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_incorrect(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([1, 0, 1, 2, 0], device=device),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 0.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 0.0,
        f"{mode}/cat_conf_mat_macro_precision": 0.0,
        f"{mode}/cat_conf_mat_macro_recall": 0.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 0.0,
        f"{mode}/cat_conf_mat_micro_precision": 0.0,
        f"{mode}/cat_conf_mat_micro_recall": 0.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 0.0,
        f"{mode}/cat_conf_mat_weighted_precision": 0.0,
        f"{mode}/cat_conf_mat_weighted_recall": 0.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 0.0,
        f"{mode}/cat_conf_mat_num_predictions": 5,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_betas(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3, betas=(0.5, 1, 2)).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f0.5_score": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_macro_f2_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f0.5_score": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_f2_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f0.5_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_f2_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 5,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_2d(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor(
            [[[3, 2, 1], [1, 3, 2]], [[3, 2, 1], [1, 2, 3]], [[1, 3, 2], [3, 2, 1]]], device=device
        ),
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("dtype_prediction", DTYPES)
@mark.parametrize("dtype_target", DTYPES)
def test_categorical_confusion_matrix_forward_dtypes(
    device: str,
    mode: str,
    dtype_prediction: torch.dtype,
    dtype_target: torch.dtype,
):
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor(
            [[[3, 2, 1], [1, 3, 2]], [[3, 2, 1], [1, 2, 3]], [[1, 3, 2], [3, 2, 1]]],
            device=device,
            dtype=dtype_prediction,
        ),
        torch.tensor([[0, 1], [0, 2], [1, 0]], device=device, dtype=dtype_target),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_multiple_batches(device: str, mode: str) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2]], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    metric(torch.tensor([[1, 2, 3], [3, 2, 1]], device=device), torch.tensor([2, 0], device=device))
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_forward_multiple_batches_with_reset(
    device: str, mode: str
) -> None:
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2]], device=device),
        torch.tensor([0, 1, 0, 1], device=device),
    )
    metric.reset()
    metric(
        torch.tensor([[1, 3, 2], [3, 2, 1], [1, 2, 3]], device=device),
        torch.tensor([1, 0, 2], device=device),
    )
    assert metric.value() == {
        f"{mode}/cat_conf_mat_accuracy": 1.0,
        f"{mode}/cat_conf_mat_balanced_accuracy": 1.0,
        f"{mode}/cat_conf_mat_macro_precision": 1.0,
        f"{mode}/cat_conf_mat_macro_recall": 1.0,
        f"{mode}/cat_conf_mat_macro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_micro_precision": 1.0,
        f"{mode}/cat_conf_mat_micro_recall": 1.0,
        f"{mode}/cat_conf_mat_micro_f1_score": 1.0,
        f"{mode}/cat_conf_mat_weighted_precision": 1.0,
        f"{mode}/cat_conf_mat_weighted_recall": 1.0,
        f"{mode}/cat_conf_mat_weighted_f1_score": 1.0,
        f"{mode}/cat_conf_mat_num_predictions": 3,
    }


@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_value_empty(mode):
    with raises(EmptyMetricError):
        CategoricalConfusionMatrix(mode, num_classes=3).value()


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_categorical_confusion_matrix_value_log_engine(device: str, mode: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(mode, num_classes=3).to(device=device)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    metric.value(engine)
    assert engine.get_history(f"{mode}/cat_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_macro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_macro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_macro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_micro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_micro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_micro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_weighted_precision").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_weighted_recall").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_weighted_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{mode}/cat_conf_mat_num_predictions").get_last_value() == 5


@mark.parametrize("device", get_available_devices())
def test_categorical_confusion_matrix_events_train(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(ct.TRAIN, num_classes=3).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.TRAIN_EPOCH_STARTED)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    engine.fire_event(EngineEvents.TRAIN_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_macro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_macro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_macro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_micro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_micro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_micro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_weighted_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_weighted_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_weighted_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.TRAIN}/cat_conf_mat_num_predictions").get_last_value() == 5


@mark.parametrize("device", get_available_devices())
def test_categorical_confusion_matrix_events_eval(device: str, engine: BaseEngine):
    device = torch.device(device)
    metric = CategoricalConfusionMatrix(ct.EVAL, num_classes=3).to(device=device)
    metric.attach(engine)
    metric(torch.zeros(4, device=device), torch.ones(4, device=device))
    engine.fire_event(EngineEvents.EVAL_EPOCH_STARTED)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]], device=device),
        torch.tensor([0, 1, 0, 1, 2], device=device),
    )
    engine.fire_event(EngineEvents.EVAL_EPOCH_COMPLETED)
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_balanced_accuracy").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_macro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_macro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_macro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_micro_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_micro_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_micro_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_weighted_precision").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_weighted_recall").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_weighted_f1_score").get_last_value() == 1.0
    assert engine.get_history(f"{ct.EVAL}/cat_conf_mat_num_predictions").get_last_value() == 5


def test_categorical_confusion_matrix_reset() -> None:
    metric = CategoricalConfusionMatrix(ct.EVAL, num_classes=3)
    metric(
        torch.tensor([[3, 2, 1], [1, 3, 2], [3, 2, 1], [1, 3, 2], [1, 2, 3]]),
        torch.tensor([0, 1, 0, 1, 2]),
    )
    assert metric._confusion_matrix.num_predictions == 5
    metric.reset()
    assert metric._confusion_matrix.num_predictions == 0
