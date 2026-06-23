"""
Trainers package

This package contains all trainers:
- BaseTrainer: Abstract base class
- RLTrainer: Reinforcement learning trainer (GRPO)
- SFTTrainer: Supervised fine-tuning trainer (to be implemented)
- PTTrainer: Continued pre-training trainer (to be implemented)
"""

from nanollm.trainers.base import BaseTrainer, TrainConfig
from nanollm.trainers.rl_trainer import RLTrainer

__all__ = [
    "BaseTrainer",
    "TrainConfig",
    "RLTrainer",
]
