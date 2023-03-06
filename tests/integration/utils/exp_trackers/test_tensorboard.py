from pathlib import Path

import numpy as np

from gravitorch.testing import tensorboard_available
from gravitorch.utils.exp_trackers.tensorboard import TensorBoardExpTracker
from gravitorch.utils.integrations import is_matplotlib_available, is_pillow_available
from gravitorch.utils.path import sanitize_path

if is_matplotlib_available():
    from matplotlib.pyplot import Figure, subplots
else:
    Figure, subplots = "matplotlib.pyplot.Figure", None  # pragma: no cover

if is_pillow_available():
    from PIL.Image import Image, fromarray
else:
    Image, fromarray = "PIL.Image.Image", None  # pragma: no cover


def create_figure() -> Figure:
    fig, axes = subplots()
    axes.imshow(np.eye(10))
    return fig


def create_image() -> Image:
    return fromarray(np.zeros((16, 16, 3), dtype=np.uint8), "RGB")


@tensorboard_available
def test_tensorboard_exp_tracker(tmp_path: Path) -> None:
    with TensorBoardExpTracker(experiment_path=tmp_path.joinpath("data")) as tracker:
        assert tracker.is_activated()
        assert not tracker.is_resumed()
        assert isinstance(tracker.artifact_path, Path)
        assert isinstance(tracker.checkpoint_path, Path)
        assert tracker.experiment_id == "fakeid0123"
        # Tag
        tracker.add_tag("key", 1)
        tracker.add_tags({"int": 1, "float": 1.12, "str": "something"})
        # Figure
        if is_matplotlib_available():
            figure = create_figure()
            tracker.log_figure("my_figure", figure)
            tracker.log_figures({"my_figure_1": figure, "my_figure_2": figure})
        # Image
        if is_pillow_available():
            image = create_image()
            tracker.log_image("my_image", image)
            tracker.log_images({"my_image_1": image, "my_image_2": image})
        # Metric
        tracker.log_metric("key", 1.2)
        tracker.log_metrics({"loss": 1.2, "accuracy": 35})

        tb_path = sanitize_path(tracker._writer.log_dir)

    assert len(list(tb_path.iterdir())) >= 1
