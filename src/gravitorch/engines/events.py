__all__ = ["EngineEvents"]


class EngineEvents:
    r"""Engine specific event names.

    Every engine should fire these events.
    """
    # Generic. These events are used in both ``train`` and ``eval``
    STARTED: str = "started"
    EPOCH_STARTED: str = "epoch_started"
    EPOCH_COMPLETED: str = "epoch_completed"
    COMPLETED: str = "completed"

    # Train
    TRAIN_STARTED: str = "train_started"
    TRAIN_EPOCH_STARTED: str = "train_epoch_started"
    TRAIN_ITERATION_STARTED: str = "train_iteration_started"
    TRAIN_FORWARD_COMPLETED: str = "train_forward_completed"
    TRAIN_BACKWARD_COMPLETED: str = "train_backward_completed"
    TRAIN_ITERATION_COMPLETED: str = "train_iteration_completed"
    TRAIN_EPOCH_COMPLETED: str = "train_epoch_completed"
    TRAIN_COMPLETED: str = "train_completed"

    # Eval
    EVAL_STARTED: str = "eval_started"
    EVAL_EPOCH_STARTED: str = "eval_epoch_started"
    EVAL_ITERATION_STARTED: str = "eval_iteration_started"
    EVAL_ITERATION_COMPLETED: str = "eval_iteration_completed"
    EVAL_EPOCH_COMPLETED: str = "eval_epoch_completed"
    EVAL_COMPLETED: str = "eval_completed"
