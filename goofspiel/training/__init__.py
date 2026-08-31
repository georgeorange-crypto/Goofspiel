"""Training pipeline package for the Goofspiel agent.

The package is importable without PyTorch.  Stage runners that actually train
neural networks import torch lazily inside their functions.
"""

from .coordinator import (
    FULL_SEQUENCE_ALIASES,
    THETA_PARENT,
    THETA_PRODUCERS,
    TrainingCoordinator,
    TrainingRunConfig,
)
from .distributed import DistributedTrainingConfig, STAGE_SEQUENCE, torchrun_command
from .checkpoint_registry import CHECKPOINT_KINDS, CheckpointRegistry
from .data import (
    AdaptiveTrajectorySample,
    ExactSample,
    FailureRecord,
    GameCorpusSample,
    JsonlStore,
    OpponentSession,
    ReanalysisRecord,
    RobustTrajectorySample,
    TeacherSample,
)

__all__ = [
    "TrainingCoordinator",
    "TrainingRunConfig",
    "THETA_PARENT",
    "THETA_PRODUCERS",
    "FULL_SEQUENCE_ALIASES",
    "DistributedTrainingConfig",
    "CHECKPOINT_KINDS",
    "CheckpointRegistry",
    "STAGE_SEQUENCE",
    "AdaptiveTrajectorySample",
    "ExactSample",
    "FailureRecord",
    "GameCorpusSample",
    "JsonlStore",
    "OpponentSession",
    "ReanalysisRecord",
    "RobustTrajectorySample",
    "TeacherSample",
    "torchrun_command",
]
