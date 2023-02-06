from pathlib import Path
from unittest.mock import Mock

from objectory import OBJECT_TARGET
from pytest import fixture, raises
from torch import nn
from torch.optim import SGD, Optimizer

from gravitorch import constants as ct
from gravitorch.creators.core import BaseCoreCreator
from gravitorch.engines import AlphaEngine, BaseEngine, EngineEvents
from gravitorch.loops.training import VanillaTrainingLoop
from gravitorch.utils.artifacts import BaseArtifact
from gravitorch.utils.engine_states import VanillaEngineState
from gravitorch.utils.evaluation_loops import VanillaEvaluationLoop
from gravitorch.utils.events import VanillaEventHandler
from gravitorch.utils.exp_trackers import BaseExpTracker, EpochStep, NoOpExpTracker
from gravitorch.utils.history import GenericHistory, MinScalarHistory
from tests.unit.engines.util import FakeDataSource

#################################
#     Tests for AlphaEngine     #
#################################


@fixture
def core_creator() -> BaseCoreCreator:
    creator = Mock(spec=BaseCoreCreator)
    model = nn.Linear(4, 6)
    creator.create.return_value = (
        FakeDataSource(),
        model,
        SGD(params=model.parameters(), lr=0.01),
        None,
    )
    return creator


def increment_epoch_handler(engine: BaseEngine) -> None:
    engine.increment_epoch(2)


def test_alpha_engine_str(core_creator: BaseCoreCreator):
    assert str(AlphaEngine(core_creator)).startswith("AlphaEngine(")


def test_alpha_engine_core_creator_dict():
    engine = AlphaEngine(
        core_creator={
            OBJECT_TARGET: "gravitorch.creators.core.AdvancedCoreCreator",
            "data_source_creator": {
                OBJECT_TARGET: "gravitorch.creators.datasource.VanillaDataSourceCreator",
                "config": {OBJECT_TARGET: "tests.unit.engines.util.FakeDataSource"},
            },
            "model_creator": {
                OBJECT_TARGET: "gravitorch.creators.model.VanillaModelCreator",
                "model_config": {
                    OBJECT_TARGET: "torch.nn.Linear",
                    "in_features": 8,
                    "out_features": 2,
                },
            },
        }
    )
    assert isinstance(engine.data_source, FakeDataSource)
    assert isinstance(engine.model, nn.Linear)


def test_alpha_engine_data_source(core_creator: BaseCoreCreator):
    assert isinstance(AlphaEngine(core_creator).data_source, FakeDataSource)


def test_alpha_engine_epoch(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).epoch == -1


def test_alpha_engine_iteration(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).iteration == -1


def test_alpha_engine_lr_scheduler(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).lr_scheduler is None


def test_alpha_engine_max_epochs(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).max_epochs == 1


def test_alpha_engine_model(core_creator: BaseCoreCreator):
    assert isinstance(AlphaEngine(core_creator).model, nn.Linear)


def test_alpha_engine_optimizer(core_creator: BaseCoreCreator):
    assert isinstance(AlphaEngine(core_creator).optimizer, Optimizer)


def test_alpha_engine_random_seed(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).random_seed == 9984043075503325450


def test_alpha_engine_should_terminate(core_creator: BaseCoreCreator):
    assert not AlphaEngine(core_creator).should_terminate


def test_alpha_engine_add_event_handler(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_event_handler(
        "my_event",
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
    )
    assert engine._event_manager.has_event_handler(
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
        "my_event",
    )


def test_alpha_engine_add_history_with_key(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_history(MinScalarHistory("name"), "loss")
    assert engine._state.has_history("loss")


def test_alpha_engine_add_history_without_key(core_creator):
    engine = AlphaEngine(core_creator)
    engine.add_history(MinScalarHistory("loss"))
    assert engine._state.has_history("loss")


def test_alpha_engine_add_module(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_module("my_module", nn.Linear(4, 6))
    assert engine._state.has_module("my_module")


def test_alpha_engine_create_artifact(core_creator: BaseCoreCreator):
    exp_tracker = Mock(spec=BaseExpTracker)
    artifact = Mock(spec=BaseArtifact)
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine.create_artifact(artifact)
    exp_tracker.create_artifact.assert_called_once_with(artifact)


def test_alpha_engine_eval(core_creator: BaseCoreCreator):
    evaluation_loop = Mock()
    engine = AlphaEngine(core_creator, evaluation_loop=evaluation_loop)
    engine.eval()
    evaluation_loop.eval.assert_called_once_with(engine)


def test_alpha_engine_fire_event(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_event_handler(
        "my_event",
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
    )
    assert engine.epoch == -1
    engine.fire_event("my_event")
    assert engine.epoch == 1


def test_alpha_engine_get_history_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    history = MinScalarHistory("loss")
    engine.add_history(history)
    assert engine.get_history("loss") is history


def test_alpha_engine_get_history_does_not_exist(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    history = engine.get_history("loss")
    assert isinstance(history, GenericHistory)
    assert len(history.get_recent_history()) == 0


def test_alpha_engine_get_histories(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    history1 = MinScalarHistory("loss")
    history2 = MinScalarHistory("accuracy")
    engine.add_history(history1)
    engine.add_history(history2)
    assert engine.get_histories() == {"loss": history1, "accuracy": history2}


def test_alpha_engine_get_histories_empty(core_creator: BaseCoreCreator):
    assert AlphaEngine(core_creator).get_histories() == {}


def test_alpha_engine_get_module_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_module("my_module", nn.Linear(4, 6))
    module = engine.get_module("my_module")
    assert isinstance(module, nn.Linear)


def test_alpha_engine_get_module_does_not_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    with raises(ValueError):
        engine.get_module("my_module")


def test_alpha_engine_has_event_handler_true(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_event_handler("my_event", VanillaEventHandler(increment_epoch_handler))
    assert engine.has_event_handler(VanillaEventHandler(increment_epoch_handler), "my_event")


def test_alpha_engine_has_event_handler_false(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert not engine.has_event_handler(VanillaEventHandler(increment_epoch_handler), "my_event")


def test_alpha_engine_has_history_true(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_history(MinScalarHistory("loss"))
    assert engine.has_history("loss")


def test_alpha_engine_has_history_false(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert not engine.has_history("loss")


def test_alpha_engine_has_module_true(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_module("my_module", nn.Linear(4, 6))
    assert engine.has_module("my_module")


def test_alpha_engine_has_module_false(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert not engine.has_module("my_module")


def test_alpha_engine_increment_epoch_1(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert engine.epoch == -1
    engine.increment_epoch()
    assert engine.epoch == 0


def test_alpha_engine_increment_epoch_2(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert engine.epoch == -1
    engine.increment_epoch(2)
    assert engine.epoch == 1


def test_alpha_engine_increment_iteration_1(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert engine.iteration == -1
    engine.increment_iteration()
    assert engine.iteration == 0


def test_alpha_engine_increment_iteration_2(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert engine.iteration == -1
    engine.increment_iteration(2)
    assert engine.iteration == 1


def test_alpha_engine_load_state_dict(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    state = engine.state_dict()
    state["epoch"] = 10
    engine.load_state_dict(state)
    assert engine.epoch == 10


def test_alpha_engine_log_figure_without_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    figure = Mock()
    engine.log_figure("loss", figure)
    exp_tracker.log_figure.assert_called_once_with("loss", figure, None)


def test_alpha_engine_log_figure_with_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    figure = Mock()
    engine.log_figure("loss", figure, step=EpochStep(3))
    exp_tracker.log_figure.assert_called_once_with("loss", figure, EpochStep(3))


def test_alpha_engine_log_figures_without_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    figure1, figure2 = Mock(), Mock()
    engine.log_figures({"figure1": figure1, "figure2": figure2})
    exp_tracker.log_figures.assert_called_once_with({"figure1": figure1, "figure2": figure2}, None)


def test_alpha_engine_log_figures_with_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    figure1, figure2 = Mock(), Mock()
    engine.log_figures({"figure1": figure1, "figure2": figure2}, step=EpochStep(3))
    exp_tracker.log_figures.assert_called_once_with(
        {"figure1": figure1, "figure2": figure2}, EpochStep(3)
    )


def test_alpha_engine_log_metric_without_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine.log_metric("loss", 0.8)
    assert engine.get_history("loss").get_last_value() == 0.8
    exp_tracker.log_metric.assert_called_once_with("loss", 0.8, None)


def test_alpha_engine_log_metric_with_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine.log_metric("loss", 0.8, step=EpochStep(3))
    assert engine.get_history("loss").get_recent_history()[-1] == (3, 0.8)
    exp_tracker.log_metric.assert_called_once_with("loss", 0.8, EpochStep(3))


def test_alpha_engine_log_metrics_without_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine.log_metrics({"loss": 0.8, "accuracy": 42})
    assert engine.get_history("loss").get_last_value() == 0.8
    assert engine.get_history("accuracy").get_last_value() == 42
    exp_tracker.log_metrics.assert_called_once_with({"loss": 0.8, "accuracy": 42}, None)


def test_alpha_engine_log_metrics_with_step(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine.log_metrics({"loss": 0.8, "accuracy": 42}, EpochStep(3))
    assert engine.get_history("loss").get_recent_history()[-1] == (3, 0.8)
    assert engine.get_history("accuracy").get_recent_history()[-1] == (3, 42)
    exp_tracker.log_metrics.assert_called_once_with({"loss": 0.8, "accuracy": 42}, EpochStep(3))


def test_alpha_engine_remove_event_handler_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_event_handler(
        "my_event",
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
    )
    engine.remove_event_handler(
        "my_event",
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
    )
    assert not engine.has_event_handler(
        VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
        "my_event",
    )


def test_alpha_engine_remove_event_handler_does_not_exist(
    core_creator: BaseCoreCreator,
):
    engine = AlphaEngine(core_creator)
    with raises(ValueError):
        engine.remove_event_handler(
            "my_event",
            VanillaEventHandler(increment_epoch_handler, handler_kwargs={"engine": engine}),
        )


def test_alpha_engine_remove_module_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.add_module("my_module", nn.Linear(4, 6))
    engine.remove_module("my_module")
    assert not engine.has_module("my_module")


def test_alpha_engine_remove_module_does_not_exists(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    with raises(ValueError):
        engine.remove_module("my_module")


def test_alpha_engine_state_dict(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    assert set(engine.state_dict().keys()) == {"modules", "epoch", "histories", "iteration"}


def test_alpha_engine_terminate_initial_false(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine.terminate()
    assert engine.should_terminate


def test_alpha_engine_terminate_initial_true(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)
    engine._should_terminate = True
    engine.terminate()
    assert engine.should_terminate


def test_alpha_engine_train(core_creator: BaseCoreCreator):
    evaluation_loop = Mock()
    training_loop = Mock()
    engine = AlphaEngine(core_creator, evaluation_loop=evaluation_loop, training_loop=training_loop)
    engine.train()
    assert engine.epoch == 0
    evaluation_loop.eval.assert_called_once_with(engine)
    training_loop.train.assert_called_once_with(engine)


def test_alpha_engine_train_max_epochs_2(core_creator: BaseCoreCreator):
    evaluation_loop = Mock()
    training_loop = Mock()
    engine = AlphaEngine(
        core_creator,
        evaluation_loop=evaluation_loop,
        training_loop=training_loop,
        state=VanillaEngineState(max_epochs=2),
    )
    engine.train()
    assert engine.epoch == 1
    assert evaluation_loop.eval.call_count == 2
    assert training_loop.train.call_count == 2


def test_alpha_engine_train_should_terminate_true(core_creator: BaseCoreCreator):
    evaluation_loop = Mock()
    training_loop = Mock()
    engine = AlphaEngine(core_creator, evaluation_loop=evaluation_loop, training_loop=training_loop)
    engine.terminate()
    engine.train()
    assert engine.epoch == -1
    evaluation_loop.eval.assert_not_called()
    training_loop.train.assert_not_called()


def test_alpha_engine_train_with_terminate(core_creator: BaseCoreCreator):
    def terminate_handler(engine: BaseEngine):
        if engine.epoch == 2:
            engine.terminate()

    evaluation_loop = Mock()
    training_loop = Mock()
    engine = AlphaEngine(
        core_creator,
        evaluation_loop=evaluation_loop,
        training_loop=training_loop,
        state=VanillaEngineState(max_epochs=10),
    )
    engine.add_event_handler(
        EngineEvents.EPOCH_COMPLETED,
        VanillaEventHandler(terminate_handler, handler_kwargs={"engine": engine}),
    )
    engine.train()
    assert engine.epoch == 2
    assert evaluation_loop.eval.call_count == 3
    assert training_loop.train.call_count == 3


def test_alpha_engine_log_best_metrics(core_creator: BaseCoreCreator):
    exp_tracker = Mock()
    engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
    engine._log_best_metrics()
    exp_tracker.log_best_metrics.assert_called_once_with({})


def test_alpha_engine_setup_exp_tracker_with_none(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)  # None is the default value of exp_tracker
    assert isinstance(engine._exp_tracker, NoOpExpTracker)


def test_alpha_engine_setup_exp_tracker_with_exp_tracker_object_not_activated(
    core_creator: BaseCoreCreator,
    tmp_path: Path,
):
    engine = AlphaEngine(core_creator, exp_tracker=NoOpExpTracker(tmp_path.as_posix()))
    assert isinstance(engine._exp_tracker, NoOpExpTracker)
    assert engine._exp_tracker.is_activated()


def test_alpha_engine_with_exp_tracker_object_activated(
    core_creator: BaseCoreCreator,
    tmp_path: Path,
):
    with NoOpExpTracker(tmp_path.as_posix()) as exp_tracker:
        engine = AlphaEngine(core_creator, exp_tracker=exp_tracker)
        assert engine._exp_tracker is exp_tracker
        assert engine._exp_tracker.is_activated()


def test_alpha_engine_setup_evaluation_loop_with_none(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)  # None is the default value of evaluation_loop
    assert isinstance(engine._evaluation_loop, VanillaEvaluationLoop)
    assert engine._evaluation_loop == engine.get_module(ct.EVALUATION_LOOP)


def test_alpha_engine_setup_evaluation_loop_with_object(core_creator: BaseCoreCreator):
    evaluation_loop = VanillaEvaluationLoop()
    engine = AlphaEngine(core_creator, evaluation_loop=evaluation_loop)
    assert engine._evaluation_loop is evaluation_loop


def test_alpha_engine_setup_training_loop_with_none(core_creator: BaseCoreCreator):
    engine = AlphaEngine(core_creator)  # None is the default value of training_loop
    assert isinstance(engine._training_loop, VanillaTrainingLoop)
    assert engine._training_loop == engine.get_module(ct.TRAINING_LOOP)


def test_alpha_engine_setup_training_loop_with_object(core_creator: BaseCoreCreator):
    training_loop = VanillaTrainingLoop()
    engine = AlphaEngine(core_creator, training_loop=training_loop)
    assert engine._training_loop is training_loop
