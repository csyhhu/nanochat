#!/usr/bin/env python3
"""
Continue pretraining (causal LM loss) on a Hugging Face causal LM, with optional
first-N-layer truncation — same idea as `scripts/chat_web.py --hf-max-layers`.

Supports periodic eval on a separate split (e.g. wikitext validation). Metrics are
written to ``trainer_state.json`` under ``--output-dir`` (train ``loss``, ``eval_loss``).

Dependencies: ``torch``, ``transformers``, ``datasets``.

Examples::

    export PYTHONPATH="$(pwd)"
    # WikiText PT + validation eval every 50 steps
    python -m scripts.qwen_continue_pt \\
      --model-id Qwen/Qwen2.5-0.5B \\
      --max-layers 6 \\
      --preset wikitext \\
      --max-samples 5000 \\
      --block-size 512 \\
      --max-steps 500 \\
      --eval-steps 50 \\
      --logging-steps 10 \\
      --output-dir ./out/qwen6-pt-wiki

    # Disable eval
    python -m scripts.qwen_continue_pt ... --no-eval
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
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    if args.preset == "wikitext":
        args.dataset = "wikitext"
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
) -> Any:
    kwargs_load: Dict[str, Any] = {"split": split}
    if dataset_config is not None:
        kwargs_load["name"] = dataset_config
    raw = load_dataset(dataset, **kwargs_load)

    if max_samples is not None and "[" not in split:
        n = min(len(raw), max_samples)
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


def _write_run_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    path = Path(output_dir) / "pt_run_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote run summary to {path}")


def main() -> None:
    args = _parse_args()
    _apply_preset(args)
    if not args.dataset:
        print("Either --preset or --dataset is required.", file=sys.stderr)
        sys.exit(2)

    use_eval = not args.no_eval
    if use_eval and not args.eval_split:
        print("Eval enabled but --eval-split is missing (use --no-eval or set --eval-split).", file=sys.stderr)
        sys.exit(2)

    try:
        from datasets import load_dataset  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        print(
            "Missing dependency. Install with e.g.\n"
            "  uv sync --group dev\n"
            "or: pip install transformers datasets\n\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    set_seed(args.seed)
    device_type = args.device_type
    torch_dtype = _resolve_torch_dtype(device_type, args.torch_dtype)
    block_size = int(args.block_size)

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

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

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
        )
        if len(eval_grouped) == 0:
            print(f"No eval rows for split {args.eval_split!r}. Disable with --no-eval.", file=sys.stderr)
            sys.exit(1)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_cuda = device_type == "cuda"
    eval_steps = int(args.eval_steps)
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
        prediction_loss_only=True,
        report_to="none",
        seed=args.seed,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        use_cpu=(device_type == "cpu"),
        use_mps_device=(device_type == "mps"),
        eval_strategy="steps" if use_eval else "no",
        eval_steps=eval_steps if use_eval else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_grouped,
        eval_dataset=eval_grouped,
        data_collator=collator,
    )

    print(
        f"Train packed blocks: {len(train_grouped)} | split={args.split!r} | "
        f"device={device_type} | max_steps={args.max_steps}"
    )
    if use_eval:
        print(
            f"Eval packed blocks: {len(eval_grouped)} | split={args.eval_split!r} | "
            f"eval_steps={eval_steps}"
        )

    initial_eval_loss: Optional[float] = None
    if use_eval:
        print("Running initial eval (step 0) …")
        initial_metrics = trainer.evaluate()
        initial_eval_loss = float(initial_metrics["eval_loss"])
        print(f"Initial eval_loss: {initial_eval_loss:.6f}")

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_train_loss = train_result.training_loss
    final_eval_loss: Optional[float] = None
    if use_eval:
        print("Running final eval …")
        final_metrics = trainer.evaluate()
        final_eval_loss = float(final_metrics["eval_loss"])
        print(f"Final eval_loss: {final_eval_loss:.6f}")

    state_path = Path(args.output_dir) / "trainer_state.json"
    summary = {
        "model_id": args.model_id,
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
        "trainer_state_json": str(state_path),
    }
    _write_run_summary(args.output_dir, summary)

    print(f"Saved model + tokenizer to {args.output_dir}")
    if state_path.exists():
        print(f"Train/eval loss history: {state_path} (log_history: loss, eval_loss)")


if __name__ == "__main__":
    main()
