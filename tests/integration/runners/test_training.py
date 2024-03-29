from objectory import OBJECT_TARGET
from pytest import mark

from gravitorch.nn.utils import state_dicts_are_equal
from gravitorch.runners.training import _run_training_pipeline


def create_engine_config(random_seed: int) -> dict:
    return {
        OBJECT_TARGET: "gravitorch.engines.AlphaEngine",
        "core_creator": {
            OBJECT_TARGET: "gravitorch.creators.core.AdvancedCoreCreator",
            "datasource_creator": {
                OBJECT_TARGET: "gravitorch.creators.datasource.DataSourceCreator",
                "config": {
                    OBJECT_TARGET: "gravitorch.datasources.DatasetDataSource",
                    "datasets": {
                        "train": {
                            OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
                            "num_examples": 20,
                            "num_classes": 6,
                            "feature_size": 12,
                            "noise_std": 0.2,
                            "random_seed": 9555372204145584096,
                        },
                        "eval": {
                            OBJECT_TARGET: "gravitorch.datasets.DummyMultiClassDataset",
                            "num_examples": 20,
                            "num_classes": 6,
                            "feature_size": 12,
                            "noise_std": 0.2,
                            "random_seed": 10447906392539197408,
                        },
                    },
                    "dataloader_creators": {
                        "train": {
                            OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator",
                            "batch_size": 4,
                            "shuffle": True,
                            "num_workers": 0,
                            "pin_memory": True,
                            "drop_last": False,
                        },
                        "eval": {
                            OBJECT_TARGET: "gravitorch.creators.dataloader.DataLoaderCreator",
                            "batch_size": 4,
                            "shuffle": False,
                            "num_workers": 0,
                            "pin_memory": True,
                            "drop_last": False,
                        },
                    },
                },
            },
            "model_creator": {
                OBJECT_TARGET: "gravitorch.creators.model.ModelCreator",
                "model_config": {
                    OBJECT_TARGET: "gravitorch.models.VanillaModel",
                    "random_seed": random_seed,
                    "network": {
                        OBJECT_TARGET: "gravitorch.models.networks.BetaMLP",
                        "input_size": 12,
                        "hidden_sizes": (20, 20, 6),
                        "activation": {OBJECT_TARGET: "torch.nn.ReLU"},
                        "dropout": 0.1,
                    },
                    "criterion": {
                        OBJECT_TARGET: "gravitorch.models.criteria.VanillaLoss",
                        "criterion": {OBJECT_TARGET: "torch.nn.CrossEntropyLoss"},
                    },
                    "metrics": {
                        "train_metric": {
                            OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                            "metric": {
                                OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                                "mode": "train",
                            },
                        },
                        "eval_metric": {
                            OBJECT_TARGET: "gravitorch.models.metrics.VanillaMetric",
                            "metric": {
                                OBJECT_TARGET: "gravitorch.models.metrics.CategoricalAccuracy",
                                "mode": "eval",
                            },
                        },
                    },
                },
            },
            "optimizer_creator": {
                OBJECT_TARGET: "gravitorch.creators.optimizer.OptimizerCreator",
                "optimizer_config": {
                    OBJECT_TARGET: "torch.optim.SGD",
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                },
            },
        },
        "training_loop": {OBJECT_TARGET: "gravitorch.loops.training.TrainingLoop"},
        "evaluation_loop": {OBJECT_TARGET: "gravitorch.loops.evaluation.EvaluationLoop"},
        "state": {
            OBJECT_TARGET: "gravitorch.utils.engine_states.EngineState",
            "max_epochs": 1,
            "random_seed": random_seed,
        },
    }


@mark.parametrize("random_seed", (42, 1))
def test_run_training_pipeline_reproducibility(random_seed: int) -> None:
    # 2 models trained with the same random seed should be identical.
    engine1 = _run_training_pipeline(
        engine=create_engine_config(random_seed),
        handlers=(),
        exp_tracker=None,
        random_seed=random_seed,
    )
    engine2 = _run_training_pipeline(
        engine=create_engine_config(random_seed),
        handlers=(),
        exp_tracker=None,
        random_seed=random_seed,
    )
    assert (
        engine1.get_history("eval/loss").get_last_value()
        == engine2.get_history("eval/loss").get_last_value()
    )
    assert state_dicts_are_equal(engine1.model, engine2.model)
