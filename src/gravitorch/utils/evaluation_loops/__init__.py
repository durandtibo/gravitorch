r"""This package contains the implementation of some evaluation loops."""

__all__ = ["VanillaEvaluationLoop", "setup_evaluation_loop"]

from gravitorch.loops.evaluation import setup_evaluation_loop
from gravitorch.loops.evaluation.vanilla import VanillaEvaluationLoop
