# This constant is used to define the input of a model.
INPUT = "input"
# This constant is used to define the target to train/evaluate a model.
TARGET = "target"
# This constant is used to define the length of a sequence.
LENGTH = "length"
# This constant is used to define the index of an example in the dataset.
INDEX = "index"
# This constant is used to define the name of an example in the dataset.
NAME = "name"
# This constant is used to define the key of the runner in the config.
RUNNER = "runner"

# Some constants related to distributed training.
DISTRIBUTED = "distributed"
LAUNCHER = "launcher"
WORKER = "worker"

# These constants are used as default name for the training and evaluation metrics.
TRAIN = "train"
EVAL = "eval"

# This constant is used to define the default loss name. This key should be in the model output.
LOSS = "loss"
PREDICTION = "prediction"
MASK = "mask"
OUTPUT = "output"

# Engine keys
CHECKPOINT = "checkpoint"
DATA = "data"
DATA_SOURCE = "data_source"
ENGINE = "engine"
EVALUATION_LOOP = "evaluation_loop"
EXP_TRACKER = "exp_tracker"
LR_SCHEDULER = "lr_scheduler"
MODEL = "model"
OPTIMIZER = "optimizer"
STATE = "state"
TRAINING_LOOP = "training_loop"
SCALER = "scaler"

EARLY_STOPPING = "early_stopping"

ARTIFACT_FOLDER_NAME = "artifacts"
CHECKPOINT_FOLDER_NAME = "checkpoints"
