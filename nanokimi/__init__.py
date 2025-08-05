"""
nanoKimi: A simplified, educational implementation of the Kimi-K2 architecture.

This package provides a clean, educational implementation of the key innovations
in Kimi-K2: Mixture of Experts (MoE), Muon optimizer, and Latent Attention.
"""

__version__ = "0.1.0"
__author__ = "nanoKimi Contributors"

from .model import KimiModel, KimiConfig
from .training import Trainer, MuonOptimizer
from .inference import Generator

__all__ = [
    "KimiModel",
    "KimiConfig", 
    "Trainer",
    "MuonOptimizer",
    "Generator",
]