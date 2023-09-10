import torch
from objectory import OBJECT_TARGET
from pytest import fixture, mark

from gravitorch import constants as ct
from gravitorch.engines import BaseEngine, EngineEvents
from gravitorch.models.metrics import (
    AbsoluteError,
    CategoricalAccuracy,
    PaddedSequenceMetric,
    VanillaMetric,
)
from gravitorch.testing import create_dummy_engine
from gravitorch.utils import get_available_devices
from gravitorch.utils.events import GEventHandler
from gravitorch.utils.history import MaxScalarHistory

SIZES = (1, 2)
MODES = (ct.TRAIN, ct.EVAL)


@fixture(scope="module")
def cri_out() -> None:
    return {}


@fixture
def engine() -> BaseEngine:
    return create_dummy_engine()


###################################
#     Tests for VanillaMetric     #
###################################


@mark.parametrize("mode", MODES)
def test_vanilla_metric_multiclass_accuracy_from_object(mode: str) -> None:
    metric = VanillaMetric(CategoricalAccuracy(mode))
    assert isinstance(metric.metric, CategoricalAccuracy)
    assert metric.metric._mode == mode


@mark.parametrize("mode", MODES)
def test_vanilla_metric_multiclass_accuracy_from_config(mode: str) -> None:
    metric = VanillaMetric(
        metric={OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy"}, mode=mode
    )
    assert isinstance(metric.metric, CategoricalAccuracy)
    assert metric.metric._mode == mode


@mark.parametrize("mode", MODES)
def test_vanilla_metric_multiclass_accuracy_from_config_with_mode(mode: str) -> None:
    metric = VanillaMetric(
        metric={OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy", "mode": mode}
    )
    assert isinstance(metric.metric, CategoricalAccuracy)
    assert metric.metric._mode == mode


def test_vanilla_metric_multiclass_accuracy_attach_train(engine: BaseEngine) -> None:
    metric = VanillaMetric(CategoricalAccuracy(ct.TRAIN))
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.TRAIN}/cat_acc_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        GEventHandler(metric.metric.reset), EngineEvents.TRAIN_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        GEventHandler(metric.metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.TRAIN_EPOCH_COMPLETED,
    )


def test_vanilla_metric_multiclass_accuracy_attach_eval(engine: BaseEngine) -> None:
    metric = VanillaMetric(CategoricalAccuracy(ct.EVAL))
    metric.attach(engine)
    assert isinstance(engine.get_history(f"{ct.EVAL}/cat_acc_accuracy"), MaxScalarHistory)
    assert engine.has_event_handler(
        GEventHandler(metric.metric.reset), EngineEvents.EVAL_EPOCH_STARTED
    )
    assert engine.has_event_handler(
        GEventHandler(metric.metric.value, handler_kwargs={"engine": engine}),
        EngineEvents.EVAL_EPOCH_COMPLETED,
    )


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
def test_vanilla_metric_multiclass_accuracy_forward_correct(
    device: str, mode: str, batch_size: int, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = VanillaMetric(CategoricalAccuracy(mode)).to(device=device)
    metric(
        cri_out,
        net_out={ct.PREDICTION: torch.eye(batch_size, device=device)},
        batch={ct.TARGET: torch.arange(batch_size, device=device)},
    )
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
def test_vanilla_metric_multiclass_accuracy_forward_incorrect(
    device: str, mode: str, batch_size: int, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = VanillaMetric(CategoricalAccuracy(mode)).to(device=device)
    prediction = torch.zeros(batch_size, 3, device=device)
    prediction[:, 0] = 1
    metric(
        cri_out,
        net_out={ct.PREDICTION: prediction},
        batch={ct.TARGET: torch.ones(batch_size, device=device)},
    )
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 0.0,
        f"{mode}/cat_acc_num_predictions": batch_size,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_vanilla_metric_multiclass_accuracy_forward_multiple_batches_with_reset(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = VanillaMetric(CategoricalAccuracy(mode)).to(device=device)
    metric(
        cri_out,
        net_out={ct.PREDICTION: torch.eye(2, device=device)},
        batch={ct.TARGET: torch.zeros(2, device=device)},
    )
    metric.reset()
    metric(
        cri_out,
        net_out={ct.PREDICTION: torch.eye(2, device=device)},
        batch={ct.TARGET: torch.tensor([0, 1], device=device)},
    )
    assert metric.value() == {
        f"{mode}/cat_acc_accuracy": 1.0,
        f"{mode}/cat_acc_num_predictions": 2,
    }


##########################################
#     Tests for PaddedSequenceMetric     #
##########################################


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_padded_sequence_metric_forward_mae_correct(
    device: str, mode: str, batch_size: int, seq_len: int, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode)).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(batch_size, seq_len, 1, device=device)},
            batch={ct.TARGET: torch.ones(batch_size, seq_len, 1, device=device)},
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": batch_size * seq_len,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
@mark.parametrize("batch_size", SIZES)
@mark.parametrize("seq_len", SIZES)
def test_padded_sequence_metric_forward_mae_incorrect(
    device: str, mode: str, batch_size: int, seq_len: int, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode)).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(batch_size, seq_len, 1, device=device)},
            batch={ct.TARGET: torch.zeros(batch_size, seq_len, 1, device=device)},
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 1.0,
        f"{mode}/abs_err_max": 1.0,
        f"{mode}/abs_err_min": 1.0,
        f"{mode}/abs_err_sum": batch_size * seq_len,
        f"{mode}/abs_err_num_predictions": batch_size * seq_len,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_padded_sequence_metric_forward_mae_correct_with_mask_true_bool(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode)).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[True, True, False, True], [True, True, True, False]],
                    dtype=torch.bool,
                    device=device,
                ),
            },
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_padded_sequence_metric_forward_mae_correct_with_mask_true_long(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode)).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.long, device=device
                ),
            },
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": 6,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_padded_sequence_metric_forward_mae_correct_with_mask_false_bool(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode), valid_value=False).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[True, True, False, True], [True, True, True, False]],
                    dtype=torch.bool,
                    device=device,
                ),
            },
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_padded_sequence_metric_forward_mae_correct_with_mask_false_long(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode), valid_value=False).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={ct.PREDICTION: torch.ones(2, 4, 1, device=device)},
            batch={
                ct.TARGET: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[1, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.long, device=device
                ),
            },
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": 2,
    }


@mark.parametrize("device", get_available_devices())
@mark.parametrize("mode", MODES)
def test_padded_sequence_metric_forward_mae_mask_in_batch_false(
    device: str, mode: str, cri_out: dict
) -> None:
    device = torch.device(device)
    metric = PaddedSequenceMetric(AbsoluteError(mode), mask_in_batch=False).to(device=device)
    assert (
        metric(
            cri_out,
            net_out={
                ct.PREDICTION: torch.ones(2, 4, 1, device=device),
                ct.MASK: torch.tensor(
                    [[True, True, False, True], [True, True, True, False]],
                    dtype=torch.bool,
                    device=device,
                ),
            },
            batch={ct.TARGET: torch.ones(2, 4, 1, device=device)},
        )
        is None
    )
    assert metric.value() == {
        f"{mode}/abs_err_mean": 0.0,
        f"{mode}/abs_err_max": 0.0,
        f"{mode}/abs_err_min": 0.0,
        f"{mode}/abs_err_sum": 0.0,
        f"{mode}/abs_err_num_predictions": 6,
    }
