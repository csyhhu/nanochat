#!/usr/bin/env python3
"""
NanoLLM — Unified training entry point for PT / SFT / RL.

Architecture
===========
- TrainConfig  : single dataclass holding all CLI args (shared + stage-specific).
- BaseTrainer  : abstract base class (nanollm/trainers/base.py).
- PTrainer      : PT stage (nanollm/trainers/pt_trainer.py).
- SFTrainer     : SFT stage (nanollm/trainers/sft_trainer.py).
- RLTrainer     : RL stage (nanollm/trainers/rl_trainer.py).
- main()        : parse CLI → get_trainer() → trainer.run().

Adding a new stage (e.g. DPO):
  1. Create nanollm/trainers/dpo_trainer.py inheriting BaseTrainer.
  2. Implement prepare_data(), _train(), _eval().
  3. Register in get_trainer().

Usage
======
  # PT
  python -m nanollm.main --stage pt --model-id Qwen/Qwen2.5-0.5B ...

  # SFT
  python -m nanollm.main --stage sft --model-id Qwen/Qwen2.5-0.5B ...

  # RL
  python -m nanollm.main --stage rl --model-id ./out/sft-model ...

  # Eval-only (works for any stage)
  python -m nanollm.main --eval-only --model-id ./out/sft-model ...
"""
import os
import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Auto-set HF_ENDPOINT for users in China or with network issues
# ---------------------------------------------------------------------------
_DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

if not os.environ.get("HF_ENDPOINT"):
    # Check if we're likely in China (by checking if huggingface.co is accessible)
    # For simplicity, we'll just set the mirror as default
    os.environ["HF_ENDPOINT"] = _DEFAULT_HF_ENDPOINT
    print(f"[Auto-config] HF_ENDPOINT not set, using default: {_DEFAULT_HF_ENDPOINT}")
    print(f"[Auto-config] To use official HuggingFace, set: $env:HF_ENDPOINT = '' (PowerShell)")
    print(f"[Auto-config] Or set your preferred mirror site.")

# ---------------------------------------------------------------------------
# Lazy imports: only import what we need after HF_HUB_OFFLINE is set.
# BaseTrainer already sets the env var at import time.
# ---------------------------------------------------------------------------
from nanollm.trainers.base import TrainConfig  # noqa: E402
from nanollm.trainers.base import BaseTrainer  # noqa: E402


# =============================================================================
# Stage registry — map --stage to Trainer class
# =============================================================================
# We import lazily so that stages that aren't used don't need their
# dependencies (e.g. you can run PT without having SFT tasks installed).
_STAGE_REGISTRY: Dict[str, str] = {
    "pt":  "nanollm.trainers.pt_trainer.PTTrainer",
    "sft": "nanollm.trainers.sft_trainer.SFTrainer",
    "rl":  "nanollm.trainers.rl_trainer.RLTrainer",
}


def get_trainer(cfg: TrainConfig) -> BaseTrainer:
    """Import the right Trainer class and instantiate it.

    Lazy import: the Trainer subclass file is only imported when needed,
    so missing dependencies for other stages don't block you.
    """
    if cfg.stage not in _STAGE_REGISTRY:
        print(f"Unknown stage: {cfg.stage}. Choose from: {list(_STAGE_REGISTRY.keys())}")
        sys.exit(1)

    module_path, class_name = _STAGE_REGISTRY[cfg.stage].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    trainer_class = getattr(module, class_name)
    return trainer_class(cfg)


# =============================================================================
# CLI argument parsing (populates TrainConfig)
# =============================================================================

def _parse_args() -> TrainConfig:
    """Parse CLI → TrainConfig dataclass."""
    p = argparse.ArgumentParser(
        description="NanoLLM: unified PT / SFT / RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
              "  python -m nanollm.main --stage pt --model-id Qwen/Qwen2.5-0.5B\n"
              "  python -m nanollm.main --stage sft --model-id Qwen/Qwen2.5-0.5B\n"
              "  python -m nanollm.main --stage rl --model-id ./out/sft-model\n"
              "  python -m nanollm.main --eval-only --model-id ./out/sft-model\n",
    )

    # --- stage ---
    p.add_argument("--stage", choices=["pt", "sft", "rl"], default="sft",
                   help="Training stage")
    p.add_argument("--eval-only", action="store_true",
                   help="Only run eval, no training")

    # --- model ---
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B",
                   help="Hub repo ID or local path")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Truncate to first N layers")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Max sequence length")
    p.add_argument("--torch-dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-type", default="cpu", choices=["cpu", "cuda", "mps"])

    # --- LoRA ---
    p.add_argument("--no-lora", action="store_true",
                   help="Disable LoRA (full finetune)")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # --- data (shared) ---
    p.add_argument("--train-tasks", default="GSM8K,SmolTalk",
                   help="Comma-separated task names for training")
    p.add_argument("--train-max-samples", type=int, default=None)
    p.add_argument("--eval-tasks", default="MMLU,GSM8K,ARC-Easy")
    p.add_argument("--eval-max-problems", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=50)

    # --- PT-specific ---
    p.add_argument("--pt-preset", default="wikitext")
    p.add_argument("--pt-block-size", type=int, default=256)
    p.add_argument("--pt-max-samples", type=int, default=5000)
    p.add_argument("--pt-eval-split", default="validation")

    # --- SFT-specific ---
    p.add_argument("--sft-no-mask-prompt", action="store_true",
                   help="Don't mask prompt tokens in SFT loss")

    # --- RL-specific ---
    p.add_argument("--rl-num-samples", type=int, default=8,
                   help="Rollouts per prompt (GRPO G)")
    p.add_argument("--rl-examples-per-step", type=int, default=4)
    p.add_argument("--rl-temperature", type=float, default=1.0)
    p.add_argument("--rl-device-batch-size", type=int, default=2,
                   help="Max sequences per forward pass (CPU-friendly)")
    
    # --- RL CPU Optimization (for CPU-only machines) ---
    p.add_argument("--rl-quantize-cpu", action="store_true",
                   help="Enable dynamic quantization for CPU (2-4x speedup)")
    p.add_argument("--rl-compile-model", action="store_true",
                   help="Enable torch.compile() for faster inference (PyTorch 2.0+)")

    # --- training (shared) ---
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # --- output ---
    p.add_argument("--output-dir", default="./out/nanollm")
    p.add_argument("--output-json", default=None,
                   help="Save eval results to JSON (eval-only mode)")
    p.add_argument("--benchmark-no-save", action="store_true",
                   help="Train but don't save checkpoints (quick test)")

    args = p.parse_args()

    # Map CLI → TrainConfig (explicit, no reflection)
    return TrainConfig(
        stage=args.stage,
        eval_only=args.eval_only,
        model_id=args.model_id,
        max_layers=args.max_layers,
        max_seq_len=args.max_seq_len,
        torch_dtype=args.torch_dtype,
        device_type=args.device_type,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_tasks=args.train_tasks,
        train_max_samples=args.train_max_samples,
        eval_tasks=args.eval_tasks,
        eval_max_problems=args.eval_max_problems,
        eval_steps=args.eval_steps,
        pt_preset=args.pt_preset,
        pt_block_size=args.pt_block_size,
        pt_max_samples=args.pt_max_samples,
        pt_eval_split=args.pt_eval_split,
        sft_mask_prompt=not args.sft_no_mask_prompt,
        rl_num_samples=args.rl_num_samples,
        rl_examples_per_step=args.rl_examples_per_step,
        rl_temperature=args.rl_temperature,
        rl_device_batch_size=args.rl_device_batch_size,
        rl_quantize_cpu=args.rl_quantize_cpu,
        rl_compile_model=args.rl_compile_model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        output_json=args.output_json,
        benchmark_no_save=args.benchmark_no_save,
    )


# =============================================================================
# main()
# =============================================================================

def main():
    cfg = _parse_args()

    print(f"\n{'='*60}")
    print(f"  NanoLLM")
    print(f"  Stage : {cfg.stage.upper()}")
    print(f"  Model : {cfg.model_id}")
    if not cfg.eval_only:
        print(f"  Steps : {cfg.max_steps}")
    print(f"{'='*60}")

    trainer = get_trainer(cfg)

    if cfg.eval_only:
        trainer.run_eval_only()
    else:
        trainer.run()


if __name__ == "__main__":
    main()
