import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gravitorch.utils.exp_trackers import (
    BaseExpTracker,
    EpochStep,
    TensorBoardExpTracker,
)


def main(tracker: BaseExpTracker) -> None:
    tracker.add_tag("my_tag", 123)

    tracker.log_hyper_parameter("my_hparam", "value")
    tracker.log_hyper_parameter("tensor", torch.ones(1, 2))
    tracker.log_hyper_parameters(
        {
            "model": {
                "network": {
                    "_target_": "MLP",
                    "input_size": 12,
                },
                "criterion": "MSE",
            },
            "dataset": "MNIST",
        }
    )

    fig, axes = plt.subplots()
    axes.imshow(np.eye(10))

    for epoch in range(10):
        tracker.log_metric("loss", 1 / (epoch + 1), EpochStep(epoch))
        tracker.log_metrics(
            {"accuracy": math.log(epoch + 1), "metric": 100 - epoch}, EpochStep(epoch)
        )

        tracker.log_figure("my_figure", fig, EpochStep(epoch))

        tracker.log_best_metric("loss", math.log(epoch + 1))
        tracker.log_best_metrics({"accuracy": math.log(epoch + 1), "metric": 2 * epoch + 1})

        tracker.flush()


if __name__ == "__main__":
    with TensorBoardExpTracker(
        os.path.join(os.getcwd(), "tmp", "tensorboard"),
    ) as tracker:
        main(tracker)
