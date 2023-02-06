r"""This package contains the implementation of some evaluation loops."""

__all__ = [
    "AMPEvaluationLoop",
    "AccelerateEvaluationLoop",
    "BaseBasicEvaluationLoop",
    "BaseEvaluationLoop",
    "NoOpEvaluationLoop",
    "VanillaEvaluationLoop",
    "setup_evaluation_loop",
]

from gravitorch.loops.evaluation import setup_evaluation_loop
from gravitorch.loops.evaluation.noop import NoOpEvaluationLoop
from gravitorch.utils.evaluation_loops.accelerate import AccelerateEvaluationLoop
from gravitorch.utils.evaluation_loops.amp import AMPEvaluationLoop
from gravitorch.utils.evaluation_loops.base import (
    BaseBasicEvaluationLoop,
    BaseEvaluationLoop,
)
from gravitorch.utils.evaluation_loops.vanilla import VanillaEvaluationLoop
