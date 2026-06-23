#!/usr/bin/env python3
# Force offline mode BEFORE any huggingface_hub / transformers import.
# AutoTokenizer.from_pretrained internally calls model_info() even when
# local_files_only=True, which triggers a connection to huggingface.co.
import os

os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT", "").strip():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
Qwen SFT (Supervised Fine-Tuning) on task datasets.

Uses HuggingFace Trainer with a custom DataCollator that masks prompt tokens
(labels = -1) so loss is only computed on assistant completions. The prompt is
rendered via Qwen's native chat_template (apply_chat_template).

Supports:
- LoRA (default) or full finetune
- Layer truncation (--max-layers)
- Customizable training data via --train-tasks
- Periodic eval via qwen_eval.py (ChatCORE tasks)
- --eval-only mode for evaluating an already-trained model

Training data sources (from tasks/):
  SmolTalk       - general conversational data (HuggingFaceTB/smol-smoltalk)
  GSM8K          - math reasoning with tool use (openai/gsm8k)
  SpellingBee    - counting letters in words
  SimpleSpelling - spelling words letter-by-letter
  CustomJSON     - custom JSONL conversation file

Eval tasks (held out, NOT used for training):
  MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SpellingBee

Examples::

    export PYTHONPATH="$(pwd)"

    # LoRA SFT with SmolTalk + GSM8K + SpellingBee
    python -m scripts.qwen_sft \
      --model-id Qwen/Qwen2.5-0.5B \
      --train-tasks SmolTalk,GSM8K,SpellingBee,SimpleSpelling \
      --max-layers 6 \
      --max-steps 500 \
      --output-dir ./out/qwen6-sft-lora

    # Quick test: only 10 steps, 10 eval problems
    python -m scripts.qwen_sft \
      --model-id Qwen/Qwen2.5-0.5B \
      --train-tasks GSM8K \
      --max-steps 10 \
      --eval-tasks MMLU \
      --eval-max-problems 10 \
      --output-dir ./out/qwen6-sft-lora-quick

    # Full finetune with all training tasks
    python -m scripts.qwen_sft \
      --model-id Qwen/Qwen2.5-0.5B \
      --train-tasks SmolTalk,GSM8K,SpellingBee,SimpleSpelling \
      --max-layers 6 \
      --full-finetune \
      --max-steps 500 \
      --output-dir ./out/qwen6-sft-full

    # Eval-only mode: just run ChatCORE eval on an existing model
    python -m scripts.qwen_sft \
      --model-id ./out/qwen6-sft-lora \
      --eval-only \
      --eval-tasks MMLU,ARC-Easy,ARC-Challenge
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from nanochat.transformers_backend import (
    TransformersChatBackend,
    _prefer_offline_hub_load,
    resolve_hf_model_path,
)

# ---------------------------------------------------------------------------
# Task registry: maps task name -> (class, default_kwargs)
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def _register():
    from tasks.smoltalk import SmolTalk
    from tasks.gsm8k import GSM8K
    from tasks.spellingbee import SpellingBee, SimpleSpelling
    from tasks.mmlu import MMLU
    from tasks.arc import ARC
    from tasks.humaneval import HumanEval

    TASK_REGISTRY.update({
        "SmolTalk":       (SmolTalk,       {"split": "train"}),
        "GSM8K":          (GSM8K,          {"subset": "main", "split": "train"}),
        "SpellingBee":    (SpellingBee,    {"size": 80000, "split": "train"}),
        "SimpleSpelling": (SimpleSpelling, {"size": 200000, "split": "train"}),
        "MMLU":           (MMLU,           {"subset": "all", "split": "auxiliary_train"}),
        "ARC-Easy":       (ARC,            {"subset": "ARC-Easy", "split": "train"}),
        "ARC-Challenge":  (ARC,            {"subset": "ARC-Challenge", "split": "train"}),
        "HumanEval":      (HumanEval,      {"split": "train"}),
    })


# ---------------------------------------------------------------------------
# Conversation -> HF messages (for Qwen chat_template)
# ---------------------------------------------------------------------------

def conversation_to_messages(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert a nanochat task conversation into a list of HF-format messages
    suitable for tokenizer.apply_chat_template().

    Handles:
    - Simple string content (SmolTalk, SimpleSpelling, MMLU, ARC)
    - Multi-part content lists (GSM8K, SpellingBee) -> flattened to text
    """
    raw_messages: List[Dict[str, Any]] = conversation["messages"]
    hf_messages: List[Dict[str, str]] = []

    for m in raw_messages:
        role = m["role"]
        content = m["content"]

        if isinstance(content, str):
            hf_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Multi-part content: flatten to a single text string
            text_parts: List[str] = []
            for part in content:
                ptype = part.get("type", "text")
                ptext = part.get("text", "")
                if ptype == "text":
                    text_parts.append(ptext)
                elif ptype == "python":
                    text_parts.append(f"<<{ptext}=")
                elif ptype == "python_output":
                    text_parts.append(f"{ptext}>>")
                else:
                    text_parts.append(str(ptext))
            hf_messages.append({"role": role, "content": "".join(text_parts)})
        else:
            hf_messages.append({"role": role, "content": str(content)})

    return hf_messages


# ---------------------------------------------------------------------------
# Dataset: wraps a Task for HF Trainer (returns tokenized conversation)
# ---------------------------------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    """
    Wraps a nanochat Task for HuggingFace Trainer.

    Each item = tokenized conversation with labels masked for non-assistant tokens.
    """

    def __init__(
        self,
        task: Any,
        tokenizer: Any,
        max_length: int,
        mask_prompt: bool = True,
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self._len = task.num_examples()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        conversation = self.task.get_example(index)
        messages = conversation_to_messages(conversation)

        # Render the FULL conversation (including assistant response) using chat_template
        # add_generation_prompt=False -> renders complete conversation
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(full_ids, torch.Tensor):
            full_ids = full_ids.squeeze(0).tolist()

        # Truncate if too long
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        input_ids = full_ids
        labels = full_ids.copy()

        if self.mask_prompt:
            # Render only the prompt (all messages EXCEPT the last assistant)
            # add_generation_prompt=True -> appends "<|im_start|>assistant\n"
            prompt_messages = messages[:-1]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.squeeze(0).tolist()

            # Mask prompt tokens: set labels to -100 (HF ignore_index)
            prompt_len = len(prompt_ids)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
            input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * (self.max_length - pad_len) + [0] * pad_len,
                dtype=torch.long,
            ),
        }


# ---------------------------------------------------------------------------
# DataCollator: standard padding collator (HF handles the rest)
# ---------------------------------------------------------------------------

class SFTDataCollator:
    """
    Simple data collator for SFT.
    Pads input_ids/labels/attention_mask to max length in batch.
    """

    def __init__(self, tokenizer: Any):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        return batch


# ---------------------------------------------------------------------------
# MixtureDataset: combines multiple SFTDatasets into one
# ---------------------------------------------------------------------------

class MixtureDataset(torch.utils.data.Dataset):
    """
    Combines multiple SFTDatasets into a single dataset by interleaving.
    Uses a deterministic shuffle (seed=42) to mix different task types.
    """

    def __init__(self, datasets: List[SFTDataset], seed: int = 42):
        self.datasets = datasets
        self.index_map: List[Tuple[int, int]] = []  # (dataset_idx, item_idx)

        for ds_idx, ds in enumerate(datasets):
            for item_idx in range(len(ds)):
                self.index_map.append((ds_idx, item_idx))

        rng = __import__("random").Random(seed)
        rng.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ds_idx, item_idx = self.index_map[index]
        return self.datasets[ds_idx][item_idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen SFT using HuggingFace Trainer on nanochat task datasets."
    )
    # Model
    p.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HF model id or local path (Base model recommended for SFT).",
    )
    p.add_argument(
        "--max-layers", type=int, default=None,
        help="Keep only first N transformer layers (e.g. 6).",
    )
    p.add_argument(
        "--max-context-len", type=int, default=2048,
        help="Clamp config/tokenizer max context length.",
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
        default="SmolTalk,GSM8K,SpellingBee,SimpleSpelling",
        help="Comma-separated task names for training (default: SmolTalk,GSM8K,SpellingBee,SimpleSpelling).",
    )
    p.add_argument(
        "--train-max-samples", type=int, default=None,
        help="Cap total training samples (after mixing). Useful for quick experiments.",
    )
    p.add_argument(
        "--max-seq-len", type=int, default=1024,
        help="Maximum sequence length for training (default: 1024).",
    )
    p.add_argument(
        "--custom-json", type=str, default=None,
        help="Path to a JSONL file for CustomJSON task (e.g. identity_conversations.jsonl).",
    )
    # Training
    p.add_argument("--max-steps", type=int, default=500, help="Optimizer steps.")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Trade compute for memory.")
    # LoRA / full finetune
    p.add_argument(
        "--full-finetune", action="store_true",
        help="Train all weights (disable LoRA). Default is LoRA-only training.",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules", type=str, default=None,
        help="Comma-separated LoRA target modules (default: Qwen attention+MLP).",
    )
    # Evaluation
    p.add_argument("--no-eval", action="store_true",
                   help="Disable evaluation during training.")
    p.add_argument(
        "--eval-tasks", type=str, default="MMLU",
        help="Comma-separated ChatCORE eval tasks (default: MMLU).",
    )
    p.add_argument("--eval-steps", type=int, default=100,
                   help="Run ChatCORE eval every N steps (only initial + final + every N steps).")
    p.add_argument("--eval-max-problems", type=int, default=200,
                   help="Cap problems per eval task (smaller = faster eval).")
    p.add_argument(
        "--eval-only", action="store_true",
        help="Run ChatCORE eval once on an already-trained model and exit.",
    )
    p.add_argument(
        "--benchmark-no-save",
        action="store_true",
        help="Skip writing model/tokenizer checkpoints (for grid benchmark runs; saves disk).",
    )
    # Output
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints and logs.",
    )
    p.add_argument(
        "--output-json", type=str, default=None,
        help="If set, write eval-only results as JSON to this path.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    return ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]


def _apply_lora(model: Any, args: argparse.Namespace) -> Any:
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    except ImportError as e:
        print(
            "LoRA requires peft. Install with:\n"
            "  pip install peft\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if args.lora_target_modules:
        targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    else:
        targets = _default_lora_targets()

    config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def _trainable_param_count(model: Any) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _create_tasks(
    task_names: List[str],
    custom_json_path: Optional[str] = None,
) -> List[Any]:
    """Create Task objects from a list of task names."""
    _register()
    tasks = []
    for name in task_names:
        name = name.strip()
        if name == "CustomJSON":
            if not custom_json_path or not os.path.isfile(custom_json_path):
                print(f"WARNING: CustomJSON requested but --custom-json not set or file not found. Skipping.")
                continue
            from tasks.customjson import CustomJSON
            tasks.append(CustomJSON(filepath=custom_json_path))
        elif name in TASK_REGISTRY:
            cls, kwargs = TASK_REGISTRY[name]
            tasks.append(cls(**kwargs))
        else:
            print(f"WARNING: Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}, CustomJSON")
    return tasks


def _run_chatcore_eval(
    model: Any,
    tokenizer: Any,
    eval_tasks: List[str],
    max_problems: int,
    device: torch.device,
    mode: str = "sft",
) -> Dict[str, float]:
    """
    Run ChatCORE evaluation by calling into qwen_eval.py's run_qwen_eval.
    Returns dict of task_name -> accuracy.
    """
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


# ---------------------------------------------------------------------------
# Eval callback: runs ChatCORE eval at specified steps during training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Loss history callback: writes train/eval loss to JSON-lines at every
# logging/eval step so we can monitor the full curve in real time.
# ---------------------------------------------------------------------------


class LossHistoryCallback(TrainerCallback):
    """Appends one JSON per line to ``output_dir/loss_history.jsonl``:

      - on_log:  {"step": N, "train_loss": X.XX, "grad_norm": X.XX}
      - on_evaluate: {"step": N, "eval_loss": X.XX}
    """

    def __init__(self, output_dir: str, filename: str = "loss_history.jsonl"):
        self._path = Path(output_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")  # truncate

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only record when there's actual training loss (not empty dict, not just epoch info)
        if not logs or "loss" not in logs:
            return
        rec: Dict[str, Any] = {"step": int(state.global_step), "train_loss": float(logs["loss"])}
        if "grad_norm" in logs:
            rec["grad_norm"] = float(logs["grad_norm"])
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or "eval_loss" not in metrics:
            return
        rec: Dict[str, Any] = {"step": int(state.global_step), "eval_loss": float(metrics["eval_loss"])}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class ChatCOREValCallback:
    """
    Callback that runs ChatCORE eval at predefined steps during training.
    Manually called after each step (not integrated into HF Trainer callback system
    because ChatCORE eval is heavy and needs model.eval() + separate device handling).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        eval_task_names: List[str],
        eval_max_problems: int,
        device: torch.device,
        eval_step_interval: int,
        max_steps: int,
        mode: str = "sft",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_task_names = eval_task_names
        self.eval_max_problems = eval_max_problems
        self.device = device
        self.eval_step_interval = eval_step_interval
        self.max_steps = max_steps
        self.mode = mode
        self.eval_steps: set = self._compute_eval_steps()

    def _compute_eval_steps(self) -> set:
        """Compute which steps to eval at."""
        steps = set()
        interval = self.eval_step_interval
        for s in range(interval, self.max_steps + 1, interval):
            steps.add(s)
        steps.add(self.max_steps)  # always eval at final step
        return steps

    def should_eval(self, step: int) -> bool:
        return step in self.eval_steps

    def run_eval(self, step: int) -> Dict[str, float]:
        print(f"\n{'=' * 60}")
        print(f"ChatCORE Eval at step {step}")
        print(f"{'=' * 60}")
        self.model.eval()
        results = _run_chatcore_eval(
            self.model, self.tokenizer, self.eval_task_names,
            max_problems=self.eval_max_problems,
            device=self.device,
            mode=self.mode,
        )
        for task_name, acc in results.items():
            print(f"  Step {step} | {task_name}: {100*acc:.2f}%")
        self.model.train()
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    eval_only = bool(args.eval_only)

    if not eval_only and args.output_dir is None:
        print("--output-dir is required for training (use --eval-only for eval-only mode).", file=sys.stderr)
        sys.exit(2)

    if eval_only and args.output_dir is None:
        args.output_dir = "."  # Trainer needs a dummy output_dir

    # HF_HUB_OFFLINE is already set at the top of this file (before imports).
    # Ensure HF_ENDPOINT is set.
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    set_seed(args.seed)
    device_type = args.device_type
    torch_dtype = _resolve_torch_dtype(device_type, args.torch_dtype)
    max_seq_len = int(args.max_seq_len)

    # ------------------------------------------------------------------
    # Load model & tokenizer
    # ------------------------------------------------------------------
    model_path = resolve_hf_model_path(args.model_id)
    model_path, local_files_only = _prefer_offline_hub_load(args.model_id, model_path)
    pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)

    print(f"Loading tokenizer from: {model_path}" + (" (local cache only)" if local_files_only else ""))
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
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params on {device}")

    # ------------------------------------------------------------------
    # Eval-only path
    # ------------------------------------------------------------------
    if eval_only:
        eval_task_names = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
        )

        print(f"\n{'=' * 60}")
        print("Eval-Only Results")
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

    # Prepare training data
    train_task_names = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    print(f"\nTraining tasks: {train_task_names}")
    task_objects = _create_tasks(train_task_names, custom_json_path=args.custom_json)

    if not task_objects:
        print("ERROR: No valid training tasks. Check --train-tasks.", file=sys.stderr)
        sys.exit(1)

    datasets = []
    for task_obj in task_objects:
        ds = SFTDataset(
            task=task_obj,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            mask_prompt=True,
        )
        datasets.append(ds)
        print(f"  {type(task_obj).__name__}: {len(ds)} examples")

    train_dataset = MixtureDataset(datasets, seed=args.seed)
    total_train = len(train_dataset)

    if args.train_max_samples and args.train_max_samples < total_train:
        indices = torch.randperm(total_train, generator=torch.Generator().manual_seed(args.seed))
        indices = indices[:args.train_max_samples].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        total_train = len(train_dataset)

    print(f"Total training examples (after mixing): {total_train}")

    # ------------------------------------------------------------------
    # LoRA or full finetune
    # ------------------------------------------------------------------
    use_lora = not args.full_finetune
    if use_lora:
        model = _apply_lora(model, args)
    else:
        print("Full finetune: all parameters trainable.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    trainable, total = _trainable_param_count(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.2f}%)")

    # ------------------------------------------------------------------
    # TrainingArguments & Trainer
    # ------------------------------------------------------------------
    use_cuda = device_type == "cuda"

    _device_kw: Dict[str, Any] = {}
    if device_type == "cpu":
        _device_kw["use_cpu"] = True
    elif device_type == "mps":
        try:
            TrainingArguments(
                output_dir="/tmp/_probe", use_mps_device=True,
                do_train=False, do_eval=False,
            )
            _device_kw["use_mps_device"] = True
        except TypeError:
            pass

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=2,
        save_strategy="no" if args.benchmark_no_save else "steps",
        prediction_loss_only=True,
        report_to="none",
        seed=args.seed,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        eval_strategy="no",
        **_device_kw,
    )

    collator = SFTDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[LossHistoryCallback(args.output_dir)],
    )

    # ------------------------------------------------------------------
    # Eval setup
    # ------------------------------------------------------------------
    eval_results: Dict[int, Dict[str, float]] = {}

    # Initial loss eval (step 0) — evaluate on training dataset before any
    # gradient updates, so the loss curve starts from step 0.
    print("\nRunning initial loss eval (step 0) ...")
    initial_metrics = trainer.evaluate(eval_dataset=train_dataset)
    initial_eval_loss = float(initial_metrics["eval_loss"])
    # Manually write to loss_history.jsonl so plot_training_loss.py sees it
    loss_path = Path(args.output_dir) / "loss_history.jsonl"
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    with open(loss_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, "eval_loss": initial_eval_loss}) + "\n")
    print(f"  Initial eval_loss: {initial_eval_loss:.6f}")

    if not args.no_eval:
        eval_task_names = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
        eval_callback = ChatCOREValCallback(
            model=model,
            tokenizer=tokenizer,
            eval_task_names=eval_task_names,
            eval_max_problems=args.eval_max_problems,
            device=device,
            eval_step_interval=args.eval_steps,
            max_steps=args.max_steps,
        )

        # Initial ChatCORE eval (step 0)
        print("\nRunning initial ChatCORE eval (step 0) ...")
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
        )
        eval_results[0] = results
        for task_name, acc in results.items():
            print(f"  Step 0 | {task_name}: {100*acc:.2f}%")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print(f"Qwen SFT: model={args.model_id}")
    print(f"  tasks={train_task_names}")
    print(f"  max_steps={args.max_steps} | max_seq_len={max_seq_len}")
    print(f"  lr={args.learning_rate} | warmup={args.warmup_steps}")
    print(f"  lora={'yes' if use_lora else 'no (full finetune)'}")
    print(f"  layers={args.max_layers}")
    print(f"  device={device_type}")
    if not args.no_eval:
        print(f"  eval: {eval_task_names} every {args.eval_steps} steps (max {args.eval_max_problems} problems)")
    print(f"{'#' * 60}")

    train_result = trainer.train()

    # Save final model (unless benchmark-no-save)
    if not args.benchmark_no_save:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nSaved model + tokenizer to {args.output_dir}")
    else:
        print(f"\nBenchmark run: skipped model/tokenizer save ({args.output_dir})")

    # ------------------------------------------------------------------
    # Final eval
    # ------------------------------------------------------------------
    if not args.no_eval:
        print("\nRunning final ChatCORE eval ...")
        model.eval()
        results = _run_chatcore_eval(
            model, tokenizer, eval_task_names,
            max_problems=args.eval_max_problems,
            device=device,
        )
        eval_results[args.max_steps] = results
        for task_name, acc in results.items():
            print(f"  Final | {task_name}: {100*acc:.2f}%")

    # Write summary
    summary = {
        "model_id": args.model_id,
        "peft": "lora" if use_lora else "none",
        "lora_rank": int(args.lora_rank) if use_lora else None,
        "lora_alpha": int(args.lora_alpha) if use_lora else None,
        "train_tasks": train_task_names,
        "trainable_params": trainable,
        "total_params": total,
        "max_layers": args.max_layers,
        "max_seq_len": max_seq_len,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "train_examples": total_train,
        "final_train_loss": train_result.training_loss,
        "eval_results": {
            str(step): {k: round(v, 4) for k, v in res.items()}
            for step, res in eval_results.items()
        } if eval_results else None,
    }
    summary_path = Path(args.output_dir) / "sft_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
