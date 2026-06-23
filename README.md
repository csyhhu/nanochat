# LLM Playground

## Introduction
This repo is forked from nanochat, with the goal of:
- Go through the whole process of LLM training (Pre-Training (PT), SFT, RL)
- Train a usable LLM model with lest cost.

## Progress
- [Standalone PT Script using Qwen2-0.5](scripts\qwen_continue_pt.py)
  - Description: It trains a k-layer Qwen2-0.5 model using initialization from origin Qwen2-0.5
  - Current Situation: Simply Training cost too much time.
  - Next Move: N.A.
  - Refer to [Pre-Training WalkThrough](tutorial\win_qwen_pre_training.md) for details.

- [Standalone SFT Script using Qwen2-0.5](scripts\qwen_sft.py)

- [Standalone RL Script using Qwen2-0.5](scripts\qwen_rl.py)
  - Refer to [Post-Training WalkThrough](tutorial\win_qwen_post_training.md) for details.

- [Unified Codes for PT/SFT/RL](./nanollm)