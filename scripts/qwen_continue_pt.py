#!/usr/bin/env python3
# Force offline mode BEFORE any huggingface_hub / transformers import.
# AutoTokenizer.from_pretrained internally calls model_info() even when
# local_files_only=True, which triggers a connection to huggingface.co.
import os

os.environ["HF_HUB_OFFLINE"] = "1"
if not os.environ.get("HF_ENDPOINT", "").strip():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
Continue pretraining (causal LM loss) on a Hugging Face causal LM, with optional
first-N-layer truncation — same idea as `scripts/chat_web.py --hf-max-layers`.

**Default: LoRA (PEFT)** on frozen base weights — a light pass over PT data, not full
finetuning. Use ``--full-finetune`` to train all parameters (old behaviour).

Supports periodic eval on a separate split (e.g. wikitext validation). Metrics are
written to ``trainer_state.json`` under ``--output-dir`` (train ``loss``, ``eval_loss``).
LoRA runs save the adapter under ``--output-dir`` (load base ``--model-id`` + adapter at inference).

Dependencies: ``torch``, ``transformers``, ``datasets``, ``accelerate``, ``peft``.

Examples::

    export PYTHONPATH="$(pwd)"
    # LoRA continue PT (default) on WikiText subset
    python -m scripts.qwen_continue_pt \\
      --model-id Qwen/Qwen2.5-0.5B \\
      --max-layers 6 \\
      --preset wikitext \\
      --max-samples 5000 \\
      --block-size 256 \\
      --max-steps 500 \\
      --output-dir ./out/qwen6-lora-wiki

    # Full finetune (all trainable weights)
    python -m scripts.qwen_continue_pt ... --full-finetune --output-dir ./out/qwen6-pt-wiki
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nanochat.transformers_backend import (
    TransformersChatBackend,
    _prefer_offline_hub_load,
    resolve_hf_model_path,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal LM continue pretraining with optional layer truncation.")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B", help="HF model id (Base recommended for PT).")
    p.add_argument("--max-layers", type=int, default=None, help="Keep only first N transformer layers (e.g. 6).")
    p.add_argument("--max-context-len", type=int, default=2048, help="Clamp config/tokenizer max length before train.")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "bfloat16", "float16"),
        help="Model load dtype. auto: float16 on MPS, else float32.",
    )
    p.add_argument("--device-type", type=str, default="cpu", choices=("cpu", "mps", "cuda"), help="Training device.")

    p.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=("wikitext",),
        help="Fill --dataset/--dataset-config/--split/--text-column (and default --eval-split).",
    )
    p.add_argument("--dataset", type=str, default=None, help="HF datasets path, e.g. HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default=None, help="Dataset config name if required.")
    p.add_argument("--split", type=str, default="train", help="Train split name or slice, e.g. train[:10000].")
    p.add_argument("--text-column", type=str, default="text", help="Column with plain text.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap train rows after load (ignored if split already sliced with [:N]).",
    )

    p.add_argument("--eval-only", action="store_true", help="Run eval once and exit (skip training, no optimizer).")
    p.add_argument("--output-json", type=str, default=None, help="If set, write eval-only results as JSON to this path.")
    p.add_argument("--no-eval", action="store_true", help="Disable evaluation during training.")
    p.add_argument(
        "--eval-split",
        type=str,
        default=None,
        help="Eval split (default for --preset wikitext: validation). Required for eval if no preset.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Cap eval rows after load (same rules as --max-samples).",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="If set, shuffle eval rows with this seed before capping. "
        "Useful for variance estimation across multiple eval runs.",
    )
    p.add_argument("--eval-steps", type=int, default=50, help="Run eval every N training steps (when eval enabled).")

    p.add_argument("--block-size", type=int, default=512, help="LM sequence length (packed blocks).")
    p.add_argument("--max-steps", type=int, default=500, help="Optimizer steps (not epochs).")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-checkpointing", action="store_true", help="Trade compute for memory.")
    p.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train all weights (disable LoRA). Default is LoRA-only training.",
    )
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (default: 16).")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32).")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated module names (default: Qwen attention+MLP projections).",
    )
    p.add_argument(
        "--benchmark-no-save",
        action="store_true",
        help="Skip writing model/tokenizer checkpoints (for grid benchmark runs; saves disk).",
    )
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (required for training; optional for --eval-only).")
    return p.parse_args()


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    if args.preset == "wikitext":
        args.dataset = "Salesforce/wikitext"
        args.dataset_config = "wikitext-103-raw-v1"
        if args.split == "train":
            args.split = "train"
        args.text_column = "text"
        if not args.no_eval and args.eval_split is None:
            args.eval_split = "validation"


def _resolve_torch_dtype(device_type: str, name: str) -> torch.dtype | None:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if device_type == "mps":
        return torch.float16
    return None


def _group_texts(examples: Dict[str, List[Any]], block_size: int) -> Dict[str, List[Any]]:
    concatenated: Dict[str, List[int]] = {k: [] for k in examples}
    for k in examples:
        for chunk in examples[k]:
            concatenated[k].extend(chunk)
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    if total_length <= 0:
        return {k: [] for k in concatenated}
    out: Dict[str, List[Any]] = {}
    for k in concatenated:
        seq = concatenated[k]
        out[k] = [seq[i : i + block_size] for i in range(0, total_length, block_size)]
    return out


def _prepare_packed_dataset(
    *,
    load_dataset: Any,
    dataset: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    block_size: int,
    tokenizer: Any,
    max_samples: Optional[int],
    desc_prefix: str,
    seed: Optional[int] = None,
) -> Any:
    try:
        from datasets.download import DownloadConfig  # type: ignore
    except ImportError:
        DownloadConfig = None

    kwargs_load: Dict[str, Any] = {"split": split}
    if dataset_config is not None:
        kwargs_load["name"] = dataset_config

    # Set a generous timeout / reduce retries so network failures don't hang forever.
    # When HF_ENDPOINT is set to a mirror that is unreachable, the default retry loop
    # can take many minutes.  We also set max_retries=1 to fail faster and let the
    # user see a clear error.
    if DownloadConfig is not None:
        kwargs_load["download_config"] = DownloadConfig(max_retries=1)
    raw = load_dataset(dataset, **kwargs_load)

    if max_samples is not None and "[" not in split:
        n = min(len(raw), max_samples)
        if seed is not None:
            raw = raw.shuffle(seed=seed).select(range(n))
        else:
            raw = raw.select(range(n))

    cols = raw.column_names
    if text_column not in cols:
        raise ValueError(f"Column {text_column!r} not in {cols} (split={split!r})")

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(batch[text_column], add_special_tokens=False)

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=cols,
        desc=f"{desc_prefix}: tokenize ({split})",
    )
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
    grouped = tokenized.map(
        lambda batch: _group_texts(batch, block_size),
        batched=True,
        desc=f"{desc_prefix}: pack ({split}, block={block_size})",
    )
    return grouped


def _default_lora_targets() -> List[str]:
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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


def _trainable_param_count(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _write_run_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    path = Path(output_dir) / "pt_run_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {path}")


def main() -> None:
    import time as time_module

    args = _parse_args()
    _apply_preset(args)
    eval_only = bool(args.eval_only)
    if not args.dataset:
        print("Either --preset or --dataset is required.", file=sys.stderr)
        sys.exit(2)

    if not eval_only and args.output_dir is None:
        print("--output-dir is required for training (use --eval-only for eval-only mode).", file=sys.stderr)
        sys.exit(2)
    if eval_only:
        # In eval-only mode, force eval on and set a default output_dir for Trainer internals.
        args.no_eval = False
        if args.output_dir is None:
            args.output_dir = "."
        if not args.eval_split:
            print("Eval-only mode requires --eval-split (or use --preset wikitext which defaults to validation).", file=sys.stderr)
            sys.exit(2)

    use_eval = not args.no_eval
    if use_eval and not args.eval_split:
        print("Eval enabled but --eval-split is missing (use --no-eval or set --eval-split).", file=sys.stderr)
        sys.exit(2)

    # HF_HUB_OFFLINE is already set at the top of this file (before imports).
    # Set HF mirror for any remaining Hub requests.
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from datasets import load_dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        print(
            "Missing dependency. Install with e.g.\n"
            "  uv sync --group dev\n"
            "or: pip install transformers datasets accelerate peft\n\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Custom callback: write train/eval loss to JSON-lines at every
    # logging/eval step so we can monitor the full curve in real time.
    # ------------------------------------------------------------------
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

    set_seed(args.seed)
    device_type = args.device_type
    torch_dtype = _resolve_torch_dtype(device_type, args.torch_dtype)
    block_size = int(args.block_size)

    # ------------------------------------------------------------------
    # Load model from local cache (offline) when possible, with HF mirror
    # already set for any remaining Hub requests (dataset download).
    # ------------------------------------------------------------------
    model_path = resolve_hf_model_path(args.model_id)
    model_path, local_files_only = _prefer_offline_hub_load(args.model_id, model_path)
    pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **pretrained_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {model_path} …" + (" (local cache only)" if local_files_only else ""))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **pretrained_kw,
    )

    if args.max_context_len:
        TransformersChatBackend._limit_context_inplace(
            model, tokenizer=tokenizer, max_context_len=int(args.max_context_len)
        )
    if args.max_layers is not None:
        TransformersChatBackend._truncate_layers_inplace(model, max_layers=int(args.max_layers))

    if args.gradient_checkpointing and not eval_only:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    use_lora = not args.full_finetune and not eval_only
    lora_targets: Optional[List[str]] = None
    if use_lora:
        lora_targets = (
            [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            if args.lora_target_modules
            else _default_lora_targets()
        )
        model = _apply_lora(model, args)
    elif not eval_only:
        print("Full finetune: all parameters trainable.", flush=True)

    trainable, total = _trainable_param_count(model)
    if not eval_only:
        print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.2f}%)", flush=True)

    train_grouped = None
    if not eval_only:
        train_grouped = _prepare_packed_dataset(
            load_dataset=load_dataset,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            text_column=args.text_column,
            block_size=block_size,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            desc_prefix="train",
        )
        if len(train_grouped) == 0:
            print("No training rows after tokenize/pack. Check --text-column / data.", file=sys.stderr)
            sys.exit(1)

    eval_grouped = None
    if use_eval:
        assert args.eval_split is not None
        eval_grouped = _prepare_packed_dataset(
            load_dataset=load_dataset,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.eval_split,
            text_column=args.text_column,
            block_size=block_size,
            tokenizer=tokenizer,
            max_samples=args.eval_max_samples,
            desc_prefix="eval",
            seed=args.eval_seed,
        )
        if len(eval_grouped) == 0:
            print(f"No eval rows for split {args.eval_split!r}. Disable with --no-eval.", file=sys.stderr)
            sys.exit(1)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_cuda = device_type == "cuda"
    eval_steps = int(args.eval_steps)

    # Build device-specific kwargs, with compatibility for transformers >= 5.x
    # which removed use_mps_device.
    _device_kw: Dict[str, Any] = {}
    if device_type == "cpu":
        _device_kw["use_cpu"] = True
    elif device_type == "mps":
        # transformers >= 5.x dropped use_mps_device; try both names.
        try:
            TrainingArguments(output_dir="/tmp/_probe", use_mps_device=True, do_train=False, do_eval=False)
            _device_kw["use_mps_device"] = True
        except TypeError:
            pass  # transformers 5.x does not accept use_mps_device

    if eval_only:
        # Minimal TrainingArguments: no optimizer/scheduler, eval only.
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
            prediction_loss_only=True,
            report_to="none",
            seed=args.seed,
            bf16=use_cuda and torch.cuda.is_bf16_supported(),
            fp16=use_cuda and not torch.cuda.is_bf16_supported(),
            do_train=False,
            do_eval=True,
            eval_strategy="no",  # manual evaluate() call
            **_device_kw,
        )
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=int(args.max_steps),
            per_device_train_batch_size=int(args.per_device_train_batch_size),
            per_device_eval_batch_size=int(args.per_device_eval_batch_size),
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
            eval_strategy="steps" if use_eval else "no",
            eval_steps=eval_steps if use_eval else None,
            **_device_kw,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_grouped,
        eval_dataset=eval_grouped,
        data_collator=collator,
        callbacks=[LossHistoryCallback(args.output_dir)] if not eval_only else None,
    )

    if not eval_only:
        print(
            f"Train packed blocks: {len(train_grouped)} | split={args.split!r} | "
            f"device={device_type} | max_steps={args.max_steps} | "
            f"peft={'lora' if use_lora else 'none'}"
        )
    if use_eval:
        print(
            f"Eval packed blocks: {len(eval_grouped)} | split={args.eval_split!r} | "
            f"eval_steps={eval_steps if not eval_only else 'N/A (eval-only)'}"
        )

    # ------------------------------------------------------------------
    # Eval-only path: run eval once, print timing, exit.
    # ------------------------------------------------------------------
    if eval_only:
        assert eval_grouped is not None
        n_blocks = len(eval_grouped)
        total_tokens = n_blocks * block_size
        n_layers = (
            len(model.model.layers)
            if hasattr(model.model, "layers")
            else "?"
        )
        print(
            f"\nEval-only mode: {n_blocks} blocks ({total_tokens:,} tokens) | "
            f"batch={args.per_device_eval_batch_size} | device={device_type} | layers={n_layers}"
        )
        print("Running eval (this may take a while on CPU) …")
        t0 = time_module.perf_counter()
        eval_metrics = trainer.evaluate()
        elapsed = time_module.perf_counter() - t0
        eval_loss = float(eval_metrics["eval_loss"])
        ppl = float(torch.exp(torch.tensor(eval_loss)).item())
        samples_per_sec = n_blocks / elapsed if elapsed > 0 else 0
        tokens_per_sec = samples_per_sec * block_size
        print(f"\n{'='*60}")
        print(f"  Eval loss          : {eval_loss:.6f}")
        print(f"  Perplexity         : {ppl:.2f}")
        print(f"  Wall-clock         : {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print(f"  Samples/second     : {samples_per_sec:.3f}")
        print(f"  Tokens/second      : {tokens_per_sec:.1f}")
        print(f"{'='*60}")

        if args.output_json:
            import json as json_module
            result = {
                "model_id": args.model_id,
                "max_layers": args.max_layers,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "eval_split": args.eval_split,
                "eval_max_samples": args.eval_max_samples,
                "block_size": block_size,
                "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
                "device_type": device_type,
                "eval_blocks": n_blocks,
                "eval_tokens": total_tokens,
                "eval_loss": round(eval_loss, 6),
                "perplexity": round(ppl, 2),
                "wall_clock_sec": round(elapsed, 1),
                "samples_per_second": round(samples_per_sec, 3),
                "tokens_per_second": round(tokens_per_sec, 1),
            }
            json_path = Path(args.output_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json_module.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Results saved to: {json_path}")
        return

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    initial_eval_loss: Optional[float] = None
    if use_eval:
        print("Running initial eval (step 0) …")
        initial_metrics = trainer.evaluate()
        initial_eval_loss = float(initial_metrics["eval_loss"])
        print(f"Initial eval_loss: {initial_eval_loss:.6f}")

    train_result = trainer.train()
    if not args.benchmark_no_save:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    final_train_loss = train_result.training_loss
    train_metrics: Dict[str, Any] = dict(getattr(train_result, "metrics", None) or {})
    # The last eval_loss was already captured by LossHistoryCallback during train(),
    # and is also available in train_result.metrics if eval happened on the last step.
    final_eval_loss: Optional[float] = None
    if use_eval and "eval_loss" in train_metrics:
        final_eval_loss = float(train_metrics["eval_loss"])

    state_path = Path(args.output_dir) / "trainer_state.json"
    summary = {
        "model_id": args.model_id,
        "peft": "lora" if use_lora else "none",
        "lora_rank": int(args.lora_rank) if use_lora else None,
        "lora_alpha": int(args.lora_alpha) if use_lora else None,
        "lora_target_modules": lora_targets,
        "trainable_params": trainable,
        "total_params": total,
        "max_layers": args.max_layers,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "train_split": args.split,
        "eval_split": args.eval_split if use_eval else None,
        "block_size": block_size,
        "max_steps": args.max_steps,
        "eval_steps": eval_steps if use_eval else None,
        "train_packed_blocks": len(train_grouped),
        "eval_packed_blocks": len(eval_grouped) if use_eval else None,
        "initial_eval_loss": initial_eval_loss,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "train_runtime_sec": train_metrics.get("train_runtime"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        "train_tokens_per_second": (
            float(train_metrics["train_samples_per_second"]) * block_size
            if train_metrics.get("train_samples_per_second") is not None
            else None
        ),
        "trainer_state_json": str(state_path) if state_path.exists() else None,
    }
    _write_run_summary(args.output_dir, summary)

    if args.benchmark_no_save:
        print(f"Benchmark run: skipped model/tokenizer save ({args.output_dir})")
    elif use_lora:
        print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")
    else:
        print(f"Saved full model + tokenizer to {args.output_dir}")
    if state_path.exists():
        print(f"Train/eval loss history: {state_path} (log_history: loss, eval_loss)")


if __name__ == "__main__":
    main()
