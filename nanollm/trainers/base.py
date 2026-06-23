#!/usr/bin/env python3
"""
BaseTrainer — abstract base class for all NanoLLM training stages.

Each stage (PT / SFT / RL) inherits from BaseTrainer and implements:
  - prepare_data() : load & tokenize datasets
  - _train()       : stage-specific training loop
  - _eval()        : ChatCORE eval (called at step 0 and final)

The template method run() orchestrates:
  setup → initial eval → prepare_data → train → final eval → save
"""

from __future__ import annotations

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import set_seed  # type: ignore

# ---------------------------------------------------------------------------
# Offline guard (must be set before any huggingface import)
# ---------------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT", "").strip():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# =============================================================================
# TrainConfig — unified config dataclass
# =============================================================================

@dataclass
class TrainConfig:
    """Unified config for all training stages.

    Shared fields are at top; stage-specific fields are grouped.
    """
    # --- stage ---
    stage: str = "sft"           # "pt" | "sft" | "rl"
    eval_only: bool = False

    # --- model ---
    model_id: str = "Qwen/Qwen2.5-0.5B"
    max_layers: Optional[int] = None
    max_seq_len: int = 512
    torch_dtype: str = "float32"   # "float32" | "float16" | "bfloat16"
    device_type: str = "cpu"

    # --- LoRA (shared across all stages) ---
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # --- data (shared) ---
    train_tasks: str = "GSM8K,SmolTalk"
    train_max_samples: Optional[int] = None
    eval_tasks: str = "MMLU,GSM8K,ARC-Easy"
    eval_max_problems: int = 50
    eval_steps: int = 50

    # --- PT-specific ---
    pt_preset: str = "wikitext"
    pt_block_size: int = 256
    pt_max_samples: int = 5000
    pt_eval_split: str = "validation"

    # --- SFT-specific ---
    sft_mask_prompt: bool = True

    # --- RL-specific ---
    rl_num_samples: int = 8           # K rollouts per prompt (GRPO G)
    rl_examples_per_step: int = 4      # batch size in examples
    rl_temperature: float = 1.0
    rl_device_batch_size: int = 2       # max sequences per forward pass
    
    # --- RL CPU Optimization (for CPU-only machines) ---
    rl_quantize_cpu: bool = False       # Enable dynamic quantization (2-4x speedup on CPU)
    rl_compile_model: bool = False      # Enable torch.compile() (PyTorch 2.0+)

    # --- training (shared) ---
    max_steps: int = 500
    batch_size: int = 2                # per-device train batch size
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42

    # --- output ---
    output_dir: str = "./out/nanollm"
    output_json: Optional[str] = None
    benchmark_no_save: bool = False

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def train_task_list(self) -> List[str]:
        return [t.strip() for t in self.train_tasks.split(",") if t.strip()]

    @property
    def eval_task_list(self) -> List[str]:
        return [t.strip() for t in self.eval_tasks.split(",") if t.strip()]

    def resolve_torch_dtype(self):
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.torch_dtype]


# =============================================================================
# BaseTrainer
# =============================================================================

class BaseTrainer(ABC):
    """Abstract base class for PT / SFT / RL trainers.

    Subclasses must implement:
      - prepare_data() : prepare train/eval datasets
      - _train()       : run the training loop, write loss_history.jsonl
      - _eval()        : run eval (ChatCORE or task-specific)

    The template method run() handles the shared flow.
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = None
        self.train_dataset = None
        self.eval_dataset = None   # PT only (SFT/RL use tasks directly)
        self.step: int = 0

        # Resolve torch dtype
        self.torch_dtype = cfg.resolve_torch_dtype()

    # ------------------------------------------------------------------
    # Setup (shared across all stages)
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Load model, tokenizer, apply truncations & LoRA."""
        set_seed(self.cfg.seed)
        print(f"\n[Setup] Loading model: {self.cfg.model_id}")

        model_path = self._resolve_model_path(self.cfg.model_id)
        self.tokenizer = self._load_tokenizer(model_path)
        self.model = self._load_model(model_path)

        if self.cfg.max_layers is not None:
            self._apply_layer_truncation(self.cfg.max_layers)

        if self.cfg.use_lora:
            self.model = self._apply_lora()

        self.device = torch.device(self.cfg.device_type)
        self.model.to(self.device)
        print(f"  Model on device: {self.device}")
        print(f"  Trainable params: {self._count_params():,}")

    # ------------------------------------------------------------------
    # Abstract methods (must override)
    # ------------------------------------------------------------------
    @abstractmethod
    def prepare_data(self) -> None:
        """Load and tokenize data for this stage.

        Sets self.train_dataset (and self.eval_dataset if applicable).
        """
        ...

    @abstractmethod
    def _train(self) -> Dict[str, Any]:
        """Run the training loop.

        Returns a summary dict with final metrics.
        """
        ...

    @abstractmethod
    def _eval(self, step: int, tag: str = "eval") -> Dict[str, float]:
        """Run eval for this stage.

        For PT: compute eval_loss on validation split.
        For SFT/RL: run ChatCORE eval (MMLU, GSM8K, etc.)

        Returns {metric_name: value}.
        """
        ...

    # ------------------------------------------------------------------
    # Template method (shared flow)
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Full training pipeline: setup → eval → data → train → eval → save.

        Returns summary dict.
        """
        t0 = time.perf_counter()

        # 1. Setup
        self.setup()

        # 2. Initial eval (step 0)
        initial_metrics = self._eval(step=0, tag="Initial")

        # 3. Prepare data
        self.prepare_data()

        # 4. Train
        train_summary = self._train()

        # 5. Final eval
        final_metrics = self._eval(step=self.cfg.max_steps, tag="Final")

        # 6. Save
        if not self.cfg.benchmark_no_save:
            self.save()
            print(f"\nModel saved to: {self.cfg.output_dir}")

        elapsed = time.perf_counter() - t0
        summary = {
            "stage": self.cfg.stage,
            "steps": self.cfg.max_steps,
            "elapsed_seconds": round(elapsed, 1),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            **train_summary,
        }
        self._write_summary(summary)
        return summary

    def run_eval_only(self) -> None:
        """Eval-only mode: load model, run eval, save results."""
        self.setup()
        self.model.eval()
        metrics = self._eval(step=0, tag="Eval-only")
        if self.cfg.output_json:
            output = {"model_id": self.cfg.model_id, "results": metrics}
            Path(self.cfg.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(self.cfg.output_json).write_text(
                json.dumps(output, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nResults saved to: {self.cfg.output_json}")

    # ------------------------------------------------------------------
    # Save (shared)
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Save model checkpoint / LoRA adapter."""
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.use_lora:
            # Save only LoRA adapter
            self.model.save_pretrained(str(output_dir / "adapter"))
            print(f"  LoRA adapter saved to {output_dir / 'adapter'}")
        else:
            self.model.save_pretrained(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        print(f"  Tokenizer saved to {output_dir}")

    # ------------------------------------------------------------------
    # Utility methods (shared, to be overridden if needed)
    # ------------------------------------------------------------------
    def _resolve_model_path(self, model_id: str) -> str:
        """Resolve model-id to a local directory.

        TODO: move to nanollm/utils/model_utils.py
        """
        raise NotImplementedError("_resolve_model_path: implement in subclass or utils")

    def _load_tokenizer(self, model_path: str):
        """Load tokenizer, guaranteed offline."""
        raise NotImplementedError("_load_tokenizer: implement in subclass or utils")

    def _load_model(self, model_path: str):
        """Load AutoModelForCausalLM."""
        raise NotImplementedError("_load_model: implement in subclass or utils")

    def _apply_layer_truncation(self, max_layers: int) -> None:
        raise NotImplementedError("_apply_layer_truncation: implement in subclass or utils")

    def _apply_lora(self):
        raise NotImplementedError("_apply_lora: implement in subclass or utils")

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _write_summary(self, summary: Dict[str, Any]) -> None:
        """Write training summary to output_dir/summary.json."""
        if self.cfg.benchmark_no_save:
            return
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSummary saved to: {summary_path}")
