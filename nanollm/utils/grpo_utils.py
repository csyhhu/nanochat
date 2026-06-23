"""
GRPO (Group Relative Policy Optimization) Utility Functions

This module contains core utility functions for the GRPO algorithm, allowing users to:
1. Clearly understand each step of GRPO
2. Easily modify GRPO implementation details
3. Experiment with different advantage calculation methods

GRPO core ideas:
1. For each prompt, generate G rollouts
2. Calculate reward for each rollout
3. Calculate group relative advantage:
   - Naive: A_i = r_i - mean(R)
   - Normalized: A_i = (r_i - mean(R)) / std(R)
4. Update using policy gradient: loss = -mean(log_prob * advantage)

Differences from PPO:
- No value network (critic)
- No KL divergence constraint
- Uses within-group relative advantage

References:
- GRPO paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
- DAPO: "Decoupled Clip and Dynamic Sampling Policy Optimization"
"""

import torch
from typing import List, Optional, Tuple


# =============================================================================
# Advantage Calculation
# =============================================================================

def compute_naive_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    Calculate naive advantage (original GRPO)
    
    Formula: A_i = r_i - mean(R)
    
    Args:
        rewards: (G,), reward for each rollout
    
    Returns:
        advantages: (G,), advantage
    
    Note:
    - This is the simplest advantage calculation method
    - No normalization, may have high variance
    """
    return rewards - rewards.mean()


def compute_normalized_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Calculate normalized advantage (optional)
    
    Formula: A_i = (r_i - mean(R)) / (std(R) + eps)
    
    Args:
        rewards: (G,), reward for each rollout
        eps: Numerical stability
    
    Returns:
        advantages: (G,), normalized advantage
    
    Note:
    - Normalization can stabilize training
    - But it's not a required step for GRPO
    """
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    
    if std_reward > 0:
        return (rewards - mean_reward) / (std_reward + eps)
    else:
        return rewards - mean_reward


def compute_grpo_advantage(
    rewards: torch.Tensor,
    method: str = "naive",
    normalize: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate GRPO advantage (unified interface)
    
    Args:
        rewards: (G,), reward for each rollout
        method: Advantage calculation method
            - "naive": A_i = r_i - mean(R)
            - "normalized": A_i = (r_i - mean(R)) / (std(R) + eps)
        normalize: Whether to apply z-score normalization to advantages
        eps: Numerical stability
    
    Returns:
        advantages: (G,), advantage
    
    Usage suggestions:
    - Beginners: Use method="naive", normalize=False (simplest)
    - Stable training: Use method="normalized", normalize=False
    - Experimental: Use normalize=True (z-score)
    """
    if method == "naive":
        advantages = compute_naive_advantage(rewards)
    elif method == "normalized":
        advantages = compute_normalized_advantage(rewards, eps=eps)
    else:
        raise ValueError(f"Unknown advantage method: {method}")
    
    # Optional z-score normalization
    if normalize:
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + eps)
    
    return advantages


# =============================================================================
# Policy Gradient Loss
# =============================================================================

def compute_policy_gradient_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate policy gradient loss
    
    Formula: loss = -mean(log_prob * advantage)
    
    Args:
        log_probs: (G, T), log probability of each token
        advantages: (G,), advantage for each sequence
        mask: (G, T), optional, 1 for valid tokens, 0 for invalid (prompt/padding)
        reduction: "mean" or "sum"
    
    Returns:
        loss: scalar tensor
    
    Note:
    - Negative sign: Because we want to maximize expected reward, so loss = -objective
    - mask: Only calculate loss for generated part, ignore prompt and padding
    """
    # Expand advantages to token level
    # advantages: (G,) -> (G, 1) -> broadcast to (G, T)
    advantages_expanded = advantages.unsqueeze(-1)
    
    # Weighted log_probs
    weighted_log_probs = log_probs * advantages_expanded
    
    # Apply mask
    if mask is not None:
        weighted_log_probs = weighted_log_probs * mask
        num_valid = mask.sum().clamp(min=1)
    else:
        num_valid = log_probs.numel()
    
    # Calculate loss
    if reduction == "mean":
        loss = -weighted_log_probs.sum() / num_valid
    elif reduction == "sum":
        loss = -weighted_log_probs.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return loss


# =============================================================================
# KL Divergence (optional, used for constraining policy updates)
# =============================================================================

def compute_kl_divergence(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate KL divergence: KL(new || old)
    
    Formula: KL = sum(exp(log_probs_old) * (log_probs_old - log_probs_new))
    
    Args:
        log_probs_new: (G, T), log probability of new policy
        log_probs_old: (G, T), log probability of old policy
        mask: (G, T), optional
    
    Returns:
        kl: scalar tensor
    
    Note:
    - Original GRPO paper does not use KL constraint
    - If you want more stable training, you can add KL constraint
    - This requires saving log_probs of the old policy (or keeping the old model)
    """
    # Calculate probabilities
    probs_old = torch.exp(log_probs_old)
    
    # KL divergence
    kl = probs_old * (log_probs_old - log_probs_new)
    
    # Apply mask
    if mask is not None:
        kl = kl * mask
        num_valid = mask.sum().clamp(min=1)
    else:
        num_valid = kl.numel()
    
    return kl.sum() / num_valid


# =============================================================================
# Rollout Generation Helper Functions
# =============================================================================

def filter_by_reward(
    sequences: List[List[int]],
    rewards: List[float],
    threshold: float = 0.0,
) -> Tuple[List[List[int]], List[float]]:
    """
    Filter rollouts by reward
    
    Args:
        sequences: List of generated sequences
        rewards: Corresponding reward list
        threshold: Reward threshold, only keep rollouts with reward > threshold
    
    Returns:
        filtered_sequences: Filtered sequences
        filtered_rewards: Filtered rewards
    
    Note:
    - Can be used to implement "only keep correct answers" training strategy
    - But this is not standard GRPO practice
    """
    filtered = [(seq, rew) for seq, rew in zip(sequences, rewards) if rew > threshold]
    
    if not filtered:
        return sequences, rewards  # If no samples meet the condition, return original data
    
    filtered_sequences, filtered_rewards = zip(*filtered)
    return list(filtered_sequences), list(filtered_rewards)


def compute_pass_at_k(
    rewards: List[List[float]],
    k: int = 4,
) -> List[float]:
    """
    Calculate pass@k metric
    
    Args:
        rewards: List[List[float]], rewards for multiple rollouts of each problem
        k: Calculate pass@k
    
    Returns:
        pass_at_k: List[float], pass@k for each problem (0.0 or 1.0)
    
    Note:
    - pass@k = 1 if at least one of the first k rollouts is correct
    - Used for evaluation, not a training metric
    """
    pass_at_k = []
    
    for problem_rewards in rewards:
        # Check if there are correct answers in the first k rollouts
        outcomes = [r > 0.0 for r in problem_rewards[:k]]
        pass_at_k.append(1.0 if any(outcomes) else 0.0)
    
    return pass_at_k


# =============================================================================
# Experimental: Advantage Calculation for Other RL Algorithms
# =============================================================================

def compute_ppo_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """
    Calculate PPO advantage (GAE)
    
    Formula: A_t = sum_{l=0}^{T-t-1} (gamma*lam)^l * delta_{t+l}
              where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    
    Args:
        rewards: (T,), reward sequence
        values: (T+1,), value function estimation (including last state's value)
        gamma: Discount factor
        lam: GAE parameter
    
    Returns:
        advantages: (T,), GAE advantage
    
    Note:
    - This is the advantage calculation method used by PPO
    - GRPO does not use this method (because it doesn't need a value network)
    - Provided here for comparison and experimentation
    """
    # Calculate TD error (delta)
    deltas = rewards + gamma * values[1:] - values[:-1]
    
    # GAE
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * gae
        advantages[t] = gae
    
    return advantages


def compute_reinforce_advantage(
    rewards: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculate REINFORCE advantage (with baseline)
    
    Formula: A_i = r_i - baseline
    
    Args:
        rewards: (G,), reward for each rollout
        baseline: scalar or (G,), baseline (if None, use mean(reward))
    
    Returns:
        advantages: (G,), advantage
    
    Note:
    - REINFORCE is the simplest form of policy gradient
    - GRPO is essentially REINFORCE with group relative baseline
    """
    if baseline is None:
        baseline = rewards.mean()
    
    return rewards - baseline
