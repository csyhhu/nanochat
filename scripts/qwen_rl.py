"""
Qwen GRPO / REINFORCE reinforcement learning script.

This script loads a Qwen SFT model (optionally with LoRA), then applies
simplified GRPO on tasks that provide a ``reward()`` method: GSM8K and
SpellingBee.  HumanEval can be added by implementing reward() in tasks/humaneval.py.

Key design decisions (same as chat_rl.py):
1. No trust region — no KL regularization to a reference model
2. On-policy — no PPO ratio+clip needed
3. Token-level normalization (DAPO style), but only (r - mu) as advantage
4. Windows CPU-friendly: single-process, small batch sizes

Usage::

    # RL on GSM8K+SpellingBee, loading SFT model from output dir
    python -m scripts.qwen_rl \\
        --model-id ./out/qwen6-sft-lora \\
        --train-tasks GSM8K,SpellingBee \\
        --max-layers 6 \\
        --max-steps 200 \\
        --output-dir ./out/qwen6-rl

    # Quick smoke test (10 steps, no save, logging every step)
    python -m scripts.qwen_rl \\
        --model-id ./out/qwen6-sft-lora \\
        --train-tasks GSM8K \\
        --max-steps 10 \\
        --logging-steps 1 \\
        --benchmark-no-save \\
        --output-dir ./checkpoints/qwen6-rl-quick

    # Eval-only on an already-trained RL model
    python -m scripts.qwen_rl \\
        --model-id ./out/qwen6-rl \\
        --eval-only \\
        --eval-tasks MMLU,GSM8K,ARC-Easy,ARC-Challenge \\
        --output-json eval_results/rl.json
"""

from __future__ import annotations

# Force offline mode BEFORE any huggingface_hub / transformers import.
# AutoTokenizer.from_pretrained internally calls model_info() even when
# local_files_only=True, which triggers a connection to huggingface.co.
import os

os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT", "").strip():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

from nanochat.transformers_backend import (
    TransformersChatBackend,
    _prefer_offline_hub_load,
    resolve_hf_model_path,
)

# ==============================================================================
# CLI arguments
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen GRPO reinforcement learning on verifiable tasks."
    )
    # Model
    p.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HF model id, or local path to a saved SFT checkpoint.",
    )
    p.add_argument(
        "--max-layers", type=int, default=None,
        help="Keep only first N transformer layers (e.g. 6).",
    )
    p.add_argument(
        "--max-context-len", type=int, default=2048,
        help="Clamp model/tokenizer max context length.",
    )
    p.add_argument(
        "--torch-dtype", type=str, default="auto",
        choices=("auto", "float32", "bfloat16", "float16"),
        help="Model load dtype. auto: float16 on MPS, else float32.",
    )
    p.add_argument(
        "--device-type", type=str, default="cpu",
        choices=("cpu", "mps", "cuda"),
        help="Training device.",
    )
    # Data
    p.add_argument(
        "--train-tasks", type=str,
        default="GSM8K,SpellingBee",
        help="Comma-separated task names with reward() support (default: GSM8K,SpellingBee).",
    )
    p.add_argument(
        "--train-split", type=str, default="train",
        help="Dataset split for training (default: train).",
    )
    p.add_argument(
        "--eval-split", type=str, default="test",
        help="Dataset split for per-step reward eval (default: test).",
    )
    # Training horizon
    p.add_argument("--max-steps", type=int, default=500,
                   help="Number of RL optimization steps.")
    p.add_argument("--num-epochs", type=int, default=1,
                   help="Number of epochs over the training data.")
    # Sampling / batch
    p.add_argument("--examples-per-step", type=int, default=4,
                   help="Training examples per optimization step.")
    p.add_argument("--num-samples", type=int, default=8,
                   help="Number of rollouts per example/question (G in GRPO).")
    p.add_argument("--device-batch-size", type=int, default=2,
                   help="Max sequences per forward pass (keep small for CPU).")
    # Generation
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max tokens to generate per sample.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature during rollouts.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-k sampling (0 = disabled).")
    # Optimization
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Learning rate (applied uniformly to all params).")
    p.add_argument("--warmup-steps", type=int, default=10,
                   help="Linear warmup steps.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Weight decay.")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Max gradient norm for clipping.")
    # LoRA
    p.add_argument("--full-finetune", action="store_true",
                   help="Train all weights (disable LoRA). Default: LoRA-only.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules", type=str, default=None,
        help="Comma-separated LoRA target modules.",
    )
    # Logging
    p.add_argument("--logging-steps", type=int, default=10,
                   help="Log training metrics every N steps.")
    # Evaluation (ChatCORE)
    p.add_argument("--no-eval", action="store_true",
                   help="Disable ChatCORE evaluation.")
    p.add_argument(
        "--eval-tasks", type=str, default="GSM8K,MMLU",
        help="Comma-separated ChatCORE eval tasks (default: GSM8K,MMLU).",
    )
    p.add_argument("--eval-steps", type=int, default=100,
                   help="Run ChatCORE eval every N steps.")
    p.add_argument("--eval-max-problems", type=int, default=200,
                   help="Cap problems per eval task.")
    p.add_argument(
        "--eval-only", action="store_true",
        help="Run ChatCORE eval once on an already-trained RL model and exit.",
    )
    # Reward eval (per-step pass@k on eval split)
    p.add_argument("--reward-eval-every", type=int, default=50,
                   help="Run reward-based eval (pass@k) every N steps.")
    p.add_argument("--reward-eval-examples", type=int, default=100,
                   help="Max examples for reward-based eval.")
    # Checkpoint / output
    p.add_argument(
        "--benchmark-no-save",
        action="store_true",
        help="Skip writing model checkpoints (for grid runs).",
    )
    p.add_argument("--save-every", type=int, default=200,
                   help="Save checkpoint every N steps.")
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints and loss_history.jsonl.",
    )
    p.add_argument(
        "--output-json", type=str, default=None,
        help="If set, write eval-only results as JSON to this path.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ==============================================================================
# Helpers
# ==============================================================================

def _resolve_torch_dtype(device_type: str, name: str) -> Optional[torch.dtype]:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if device_type == "mps":
        return torch.float16
    return None


def _default_lora_targets() -> List[str]:
    """Return default LoRA target modules for Qwen2.5."""
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def _apply_lora(model: Any, args: argparse.Namespace) -> Any:
    """Apply LoRA to the Qwen model using peft."""
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError:
        print(
            "peft not installed. Install with: pip install peft\n"
            "Or use --full-finetune to skip LoRA.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_modules = (
        [m.strip() for m in args.lora_target_modules.split(",")]
        if args.lora_target_modules
        else _default_lora_targets()
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")
    print(f"  rank={args.lora_rank}, alpha={args.lora_alpha}, targets={target_modules}")
    return model


# ==============================================================================
# Task loading
# ==============================================================================

def _create_rl_task(task_name: str, split: str = "train") -> Any:
    """
    Create a task object that supports reward().
    RL tasks must implement reward(conversation, assistant_response) -> float.
    """
    if task_name == "GSM8K":
        from tasks.gsm8k import GSM8K
        return GSM8K(subset="main", split=split)
    elif task_name == "SpellingBee":
        from tasks.spellingbee import SpellingBee
        return SpellingBee(size=256, split=split)
    else:
        raise ValueError(
            f"Task '{task_name}' does not support reward(). "
            f"Available RL tasks: GSM8K, SpellingBee"
        )


# ==============================================================================
# QwenEngine for RL rollouts
# ==============================================================================

class QwenRLEngine:
    """
    Minimal generation engine for Qwen RL rollouts.
    Generates num_samples completions for a given prompt.

    Unlike nanochat's Engine (which supports tool-use state machines),
    this engine uses simple HF .generate() for compatibility with Qwen.
    The model is expected to output its own reasoning + answer,
    and we use GSM8K's #### <number> format or SpellingBee's
    extract_answer to verify correctness.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: List[int],
        num_samples: int = 1,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate num_samples completions.

        Returns:
            generated_sequences: list of full token sequences (prompt + generation)
            masks: list of masks (1 = generated, 0 = prompt) — same length as sequences
        """
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = input_ids.shape[-1]

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample and top_k > 0 else None,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_sequences = []
        masks = []
        for i in range(num_samples):
            if seed is not None:
                torch.manual_seed(seed + i)
            gen = self.model.generate(input_ids=input_ids, **gen_kwargs)
            seq = gen[0].tolist()
            generated_sequences.append(seq)
            # mask: 0 for prompt, 1 for generated tokens
            mask = [0] * prompt_len + [1] * (len(seq) - prompt_len)
            masks.append(mask)

        return generated_sequences, masks


# ==============================================================================
# Tokenizer wrapper for RL rendering
# ==============================================================================

class QwenRLTokenizer:
    """
    Wraps a Qwen tokenizer to provide render_for_completion() for RL.
    Removes the last assistant message and appends the generation prompt.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def render_for_completion(self, conversation: Dict[str, Any]) -> List[int]:
        """
        Render conversation, removing the last assistant message
        and adding generation prompt.
        """
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()

        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist()
            return ids

        # Fallback
        text_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)
        return self.tokenizer.encode(text, add_special_tokens=False)


# ==============================================================================
# ChatCORE eval (reuse qwen_eval.py)
# ==============================================================================

def _run_chatcore_eval(
    model: Any,
    tokenizer: Any,
    eval_tasks: List[str],
    max_problems: int,
    device: torch.device,
    mode: str = "rl",
) -> Dict[str, float]:
    """Run ChatCORE evaluation by calling into qwen_eval.py's run_qwen_eval."""
    try:
        from scripts.qwen_eval import run_qwen_eval
    except ImportError:
        print("WARNING: Cannot import qwen_eval. Skipping ChatCORE eval.", file=sys.stderr)
        return {}

    print(f"\n{'=' * 60}")
    print(f"ChatCORE Eval: {eval_tasks} (max {max_problems} problems each)")
    print(f"{'=' * 60}")

    results = run_qwen_eval(
        task_names=eval_tasks,
        model=model,
        tokenizer=tokenizer,
        mode=mode,
        batch_size=4,
        num_samples=1,
        max_new_tokens=256,
        temperature=0.0,
        top_k=50,
        max_problems=max_problems,
    )
    return results


# ==============================================================================
# Loss/reward history writer
# ==============================================================================

class RLHistoryWriter:
    """Writes training metrics to a JSONL file for plotting."""

    def __init__(self, output_dir: str, filename: str = "loss_history.jsonl"):
        self._path = Path(output_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")  # truncate

    def write(self, record: Dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ==============================================================================
# Per-step reward eval (pass@k)
# ==============================================================================

@torch.no_grad()
def _run_reward_eval(
    task: Any,
    rl_tokenizer: QwenRLTokenizer,
    engine: QwenRLEngine,
    max_examples: int = 100,
    k: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_k: int = 50,
) -> Dict[str, float]:
    """
    Evaluate pass@k on the given task using reward().
    Returns dict with pass@{1..k} metrics.
    """
    num_examples = min(max_examples, len(task))
    passk_counts = [0] * k  # pass@1, pass@2, ..., pass@k

    for idx in range(num_examples):
        conversation = task[idx]
        prompt_ids = rl_tokenizer.render_for_completion(conversation)
        prefix_len = len(prompt_ids)

        seqs, _masks = engine.generate_batch(
            prompt_ids,
            num_samples=k,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=42 + idx,
        )

        outcomes = []
        for seq in seqs:
            gen_tokens = seq[prefix_len:]
            gen_text = engine.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            reward = task.reward(conversation, gen_text)
            outcomes.append(reward > 0.0)

        # pass@i = at least one correct in first i samples
        for i in range(1, k + 1):
            if any(outcomes[:i]):
                passk_counts[i - 1] += 1

    results = {}
    for i in range(1, k + 1):
        results[f"pass@{i}"] = passk_counts[i - 1] / num_examples
    return results


# ==============================================================================
# Rollout generator
# ==============================================================================

def _make_rollout_iterator(
    train_task: Any,
    rl_tokenizer: QwenRLTokenizer,
    engine: QwenRLEngine,
    args: argparse.Namespace,
    device: torch.device,
):
    """
    Infinite iterator that yields batches for RL training.
    Each yield is: (inputs, targets, rewards, advantages)
    """
    num_examples = len(train_task)
    assistant_end = engine.tokenizer.eos_token_id

    for example_idx in itertools.cycle(range(num_examples)):
        # Get the conversation
        conversation = train_task[example_idx]

        # Render prompt (remove last assistant, add generation prompt)
        tokens = rl_tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples rollouts
        engine.model.eval()
        generated_sequences = []
        masks = []
        num_sampling_steps = max(1, args.num_samples // args.device_batch_size)
        for sampling_step in range(num_sampling_steps):
            seed = hash((example_idx, sampling_step)) & 0x7FFFFFFF
            seqs_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
            )
            generated_sequences.extend(seqs_batch)
            masks.extend(masks_batch)

        # Trim to exact num_samples
        generated_sequences = generated_sequences[:args.num_samples]
        masks = masks[:args.num_samples]

        # Calculate rewards
        rewards = []
        for sample_tokens in generated_sequences:
            gen_tokens = sample_tokens[prefix_length:]
            gen_text = engine.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            reward = train_task.reward(conversation, gen_text)
            rewards.append(reward)

        # Pad sequences to same length
        max_len = max(len(seq) for seq in generated_sequences)
        padded_seqs = [seq + [assistant_end] * (max_len - len(seq)) for seq in generated_sequences]
        padded_masks = [m + [0] * (max_len - len(m)) for m in masks]

        ids = torch.tensor(padded_seqs, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        # Build inputs / targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        # Ignore loss on prompt tokens and padding (mask=0)
        targets[mask_ids[:, 1:] == 0] = -1

        rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)

        # Advantage: reward minus mean (no z-score normalization)
        mu = rewards_t.mean()
        advantages = rewards_t - mu

        yield inputs, targets, rewards_t, advantages


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    args = _parse_args()
    eval_only = bool(args.eval_only)

    if not eval_only and args.output_dir is None:
        print("--output-dir is required for training (use --eval-only for eval-only mode).",
              file=sys.stderr)
        sys.exit(2)

    if eval_only and args.output_dir is None:
        args.output_dir = "."

    # HF_HUB_OFFLINE is already set at the top of this file (before imports).
    # Ensure HF_ENDPOINT is set (in case the top-level setting was overridden).
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    set_seed(args.seed)
    device_type = args.device_type
    torch_dtype = _resolve_torch_dtype(device_type, args.torch_dtype)

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    model_path = resolve_hf_model_path(args.model_id)
    model_path, local_files_only = _prefer_offline_hub_load(args.model_id, model_path)
    pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)

    print(f"Loading tokenizer from: {model_path}" +
          (" (local cache only)" if local_files_only else ""))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **pretrained_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **pretrained_kw,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Apply truncations
    if args.max_context_len:
        TransformersChatBackend._limit_context_inplace(
            model, tokenizer=tokenizer, max_context_len=int(args.max_context_len)
        )
    if args.max_layers is not None:
        TransformersChatBackend._truncate_layers_inplace(model, max_layers=int(args.max_layers))
        print(f"Truncated to {args.max_layers} layers")

    device = torch.device(device_type)
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params on {device}")

    # ------------------------------------------------------------------
    # Eval-only path
    # ------------------------------------------------------------------
    if eval_only:
        model.eval()
        eval_task_names = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
            mode="rl",
        )

        print(f"\n{'=' * 60}")
        print("Eval-Only Results (RL model)")
        print(f"{'=' * 60}")
        for task_name, acc in results.items():
            print(f"  {task_name:<20} {100*acc:.2f}%")

        if args.output_json:
            output = {
                "model_id": args.model_id,
                "max_layers": args.max_layers,
                "device_type": device_type,
                "eval_tasks": eval_task_names,
                "max_problems": args.eval_max_problems,
                "results": {k: round(v, 4) for k, v in results.items()},
            }
            json_path = Path(args.output_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Results saved to: {json_path}")
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    # Apply LoRA (or full finetune)
    use_lora = not args.full_finetune
    if use_lora:
        model = _apply_lora(model, args)

    # Set up training tasks
    train_task_names = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    print(f"\nRL Training tasks: {train_task_names}")

    # Load tasks and build a round-robin mixture
    train_tasks = []
    for tname in train_task_names:
        task = _create_rl_task(tname, split=args.train_split)
        train_tasks.append(task)
        print(f"  {tname}: {len(task)} examples (split={args.train_split})")

    # Build a cyclic iterator over all tasks
    # Simple approach: interleave tasks round-robin
    task_iterators = [itertools.cycle(range(len(t))) for t in train_tasks]
    num_steps_per_epoch = sum(len(t) for t in train_tasks) // args.examples_per_step
    num_steps = num_steps_per_epoch * args.num_epochs
    print(f"Estimated steps: {num_steps} (over {args.num_epochs} epoch(s))")

    # Override with --max-steps
    num_steps = min(num_steps, int(args.max_steps))

    # Build the RL tokenizer and engine
    rl_tokenizer = QwenRLTokenizer(tokenizer)
    engine = QwenRLEngine(model, tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    # LR scheduler: linear warmup then constant
    def get_lr_multiplier(step: int) -> float:
        warmup = int(args.warmup_steps)
        if warmup > 0 and step < warmup:
            return step / warmup
        return 1.0

    # History writer
    history_writer = RLHistoryWriter(args.output_dir)

    # ------------------------------------------------------------------
    # Initial ChatCORE eval (step 0)
    # ------------------------------------------------------------------
    eval_task_names = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    eval_results: Dict[int, Dict[str, float]] = {}

    if not args.no_eval:
        print("\nRunning initial ChatCORE eval (step 0) ...")
        model.eval()
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
            mode="rl",
        )
        eval_results[0] = results
        for task_name, acc in results.items():
            print(f"  Step 0 | {task_name}: {100*acc:.2f}%")

    # ------------------------------------------------------------------
    # Initial reward eval (step 0)
    # ------------------------------------------------------------------
    if args.reward_eval_every > 0:
        print("\nRunning initial reward eval (step 0) ...")
        model.eval()
        for tname in train_task_names:
            eval_task = _create_rl_task(tname, split=args.eval_split)
            reward_metrics = _run_reward_eval(
                eval_task, rl_tokenizer, engine,
                max_examples=min(args.reward_eval_examples, len(eval_task)),
                k=min(4, args.num_samples),
                max_new_tokens=args.max_new_tokens,
                temperature=0.6,
            )
            for metric_name, value in reward_metrics.items():
                print(f"  Step 0 | {tname} {metric_name}: {value:.4f}")

    # ------------------------------------------------------------------
    # Training loop header
    # ------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print(f"Qwen RL: model={args.model_id}")
    print(f"  tasks={train_task_names}")
    print(f"  steps={num_steps} | examples_per_step={args.examples_per_step}")
    print(f"  num_samples={args.num_samples} | temperature={args.temperature}")
    print(f"  lr={args.learning_rate} | warmup={args.warmup_steps}")
    print(f"  lora={'yes' if use_lora else 'no (full finetune)'}")
    print(f"  layers={args.max_layers} | device={device_type}")
    if not args.no_eval:
        print(f"  ChatCORE eval: {eval_task_names} every {args.eval_steps} steps")
    if args.reward_eval_every > 0:
        print(f"  Reward eval: every {args.reward_eval_every} steps (pass@k)")
    print(f"{'#' * 60}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_step = 0

    # Build a single rollout iterator that cycles through all tasks
    # We'll alternate between tasks using a round-robin scheme
    rollout_iters = [
        _make_rollout_iterator(task, rl_tokenizer, engine, args, device)
        for task in train_tasks
    ]
    task_idx = 0

    while train_step < num_steps:
        # Pick next task (round-robin)
        current_task = train_tasks[task_idx % len(train_tasks)]
        current_iter = rollout_iters[task_idx % len(train_tasks)]
        task_idx += 1

        # Accumulate rollouts over examples_per_step examples
        all_rewards = []
        for ex_step in range(args.examples_per_step):
            inputs, targets, rewards, advantages = next(current_iter)

            # Forward/backward in micro-batches
            model.train()
            bs = inputs.size(0)
            num_passes = max(1, bs // args.device_batch_size)

            for pass_idx in range(num_passes):
                b0 = pass_idx * args.device_batch_size
                b1 = min(b0 + args.device_batch_size, bs)
                inp = inputs[b0:b1]
                tgt = targets[b0:b1]
                adv = advantages[b0:b1]

                # Forward: get log-probabilities
                # HF model returns loss when labels are provided
                # We need per-token log probs, so we compute loss with reduction='none'
                outputs = model(inp, labels=tgt)
                # loss is averaged over non-ignored tokens; we need per-token NLL
                # Use a different approach: get logits, compute log_softmax, gather
                with torch.no_grad():
                    # Get logits without loss computation
                    logits = model(inp).logits  # (B, T, V)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # Gather log_probs at target token positions
                # tgt has -1 for ignored positions
                gathered = torch.gather(
                    log_probs, dim=-1,
                    index=tgt.clamp(min=0).unsqueeze(-1)
                ).squeeze(-1)  # (B, T)
                # Mask ignored positions
                valid_mask = (tgt >= 0).float()
                gathered = gathered * valid_mask

                # Policy gradient objective
                pg_obj = (gathered * adv.unsqueeze(-1)).sum()
                num_valid = valid_mask.sum().clamp(min=1)
                pg_obj = pg_obj / (num_valid * num_passes * args.examples_per_step)

                loss = -pg_obj
                loss.backward()

            all_rewards.append(rewards.mean().item())

        # Gradient clipping and update
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(args.max_grad_norm),
            )

        lrm = get_lr_multiplier(train_step)
        for group in optimizer.param_groups:
            group["lr"] = float(args.learning_rate) * lrm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if train_step % args.logging_steps == 0 or train_step == num_steps - 1:
            history_writer.write({
                "step": train_step,
                "mean_reward": round(mean_reward, 6),
                "lr": round(float(args.learning_rate) * lrm, 8),
            })
            print(f"Step {train_step}/{num_steps} | "
                  f"mean_reward={mean_reward:.4f} | "
                  f"lr={float(args.learning_rate) * lrm:.2e}")

        # ------------------------------------------------------------------
        # Reward eval (pass@k)
        # ------------------------------------------------------------------
        if (args.reward_eval_every > 0 and
                train_step > 0 and
                train_step % args.reward_eval_every == 0):
            print(f"\n--- Reward eval at step {train_step} ---")
            model.eval()
            for tname in train_task_names:
                eval_task = _create_rl_task(tname, split=args.eval_split)
                reward_metrics = _run_reward_eval(
                    eval_task, rl_tokenizer, engine,
                    max_examples=min(args.reward_eval_examples, len(eval_task)),
                    k=min(4, args.num_samples),
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.6,
                )
                for metric_name, value in reward_metrics.items():
                    history_writer.write({
                        "step": train_step,
                        f"{tname}_{metric_name}": round(value, 6),
                    })
                    print(f"  Step {train_step} | {tname} {metric_name}: {value:.4f}")

        # ------------------------------------------------------------------
        # ChatCORE eval
        # ------------------------------------------------------------------
        if not args.no_eval and train_step > 0 and train_step % args.eval_steps == 0:
            print(f"\n--- ChatCORE eval at step {train_step} ---")
            model.eval()
            results = _run_chatcore_eval(
                model, tokenizer, eval_task_names,
                max_problems=args.eval_max_problems,
                device=device,
                mode="rl",
            )
            eval_results[train_step] = results
            for task_name, acc in results.items():
                print(f"  Step {train_step} | {task_name}: {100*acc:.2f}%")

        # ------------------------------------------------------------------
        # Checkpoint save
        # ------------------------------------------------------------------
        if (not args.benchmark_no_save and
                train_step > 0 and
                (train_step % args.save_every == 0 or train_step == num_steps - 1)):
            save_dir = Path(args.output_dir) / f"checkpoint-{train_step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"✅ Saved checkpoint to {save_dir}")

        train_step += 1

    # ------------------------------------------------------------------
    # Final ChatCORE eval
    # ------------------------------------------------------------------
    if not args.no_eval:
        print(f"\n--- Final ChatCORE eval at step {num_steps} ---")
        model.eval()
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
            mode="rl",
        )
        eval_results[num_steps] = results
        for task_name, acc in results.items():
            print(f"  Step {num_steps} | {task_name}: {100*acc:.2f}%")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    if not args.benchmark_no_save:
        save_dir = Path(args.output_dir) / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"\n✅ Saved final model to {save_dir}")
    else:
        print(f"\nBenchmark run: skipped model save ({args.output_dir})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RL Training Complete")
    print(f"{'=' * 60}")
    print(f"  Total steps: {train_step}")
    print(f"  Output: {args.output_dir}")
    print(f"  Loss history: {args.output_dir}/loss_history.jsonl")
    if eval_results:
        print(f"\n  ChatCORE results:")
        for step, results in eval_results.items():
            for task_name, acc in results.items():
                print(f"    Step {step:>5} | {task_name}: {100*acc:.2f}%")


if __name__ == "__main__":
    main()
