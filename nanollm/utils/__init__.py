"""
Utils package

This package contains various utility functions:
- grpo_utils: GRPO algorithm utility functions
- model_utils: Model loading, saving, and other utilities (to be implemented)
- data_utils: Data processing utilities (to be implemented)
"""

from nanollm.utils.grpo_utils import (
    compute_grpo_advantage,
    compute_policy_gradient_loss,
    compute_naive_advantage,
    compute_normalized_advantage,
)

__all__ = [
    "compute_grpo_advantage",
    "compute_policy_gradient_loss",
    "compute_naive_advantage",
    "compute_normalized_advantage",
]
