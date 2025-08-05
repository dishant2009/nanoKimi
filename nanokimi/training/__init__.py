"""Training components for nanoKimi."""

from .trainer import Trainer, TrainingConfig, create_trainer
from .optimizer import MuonOptimizer
from .data import TokenDataset, DataConfig, create_datasets
from .utils import get_lr_scheduler, compute_loss
from .distributed import (
    DistributedConfig, DistributedTrainingManager, 
    setup_distributed_training, launch_distributed_training
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "create_trainer",
    "MuonOptimizer",
    "TokenDataset",
    "DataConfig",
    "create_datasets",
    "get_lr_scheduler",
    "compute_loss",
    "DistributedConfig",
    "DistributedTrainingManager",
    "setup_distributed_training",
    "launch_distributed_training",
]