#!/usr/bin/env python3
"""
Continue pretraining (causal LM loss) on a Hugging Face causal LM, with optional
first-N-layer truncation — same idea as `scripts/chat_web.py --hf-max-layers`.

Typical Mac workflow: small `--block-size`, modest `--max-steps`, `--device-type cpu`
(stable for Qwen2.5; MPS may work for tiny batches but is not guaranteed).

Dependencies (often in a separate env from full nanochat): `torch`, `transformers`,
`datasets`. Project already lists `datasets`; `transformers` is in the dev group.

Examples::

    export PYTHONPATH="$(pwd)"
    # Smoke (tiny data, 1 step)
    python -m scripts.qwen_continue_pt \\
      --model-id Qwen/Qwen2.5-0.5B \\
      --max-layers 6 \\
      --preset wikitext \\
      --max-samples 500 \\
      --block-size 256 \\
      --max-steps 2 \\
      --output-dir ./out/qwen6-pt-smoke

    # More text (still small); FineWeb-Edu sample needs network + HF acceptance
    python -m scripts.qwen_continue_pt \\
      --model-id Qwen/Qwen2.5-0.5B \\
      --max-layers 6 \\
      --dataset HuggingFaceFW/fineweb-edu \\
      --dataset-config sample-10BT \\
      --split train[:20000] \\
      --text-column text \\
      --block-size 512 \\
      --max-steps 200 \\
      --output-dir ./out/qwen6-pt-fw
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List

import torch

from nanochat.transformers_backend import TransformersChatBackend


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
        help="Fill --dataset/--dataset-config/--split/--text-column for a small public corpus.",
    )
    p.add_argument("--dataset", type=str, default=None, help="HF datasets path, e.g. HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default=None, help="Dataset config name if required.")
    p.add_argument("--split", type=str, default="train", help="Split name or slice, e.g. train[:10000].")
    p.add_argument("--text-column", type=str, default="text", help="Column with plain text.")
    p.add_argument("--max-samples", type=int, default=None, help="After load, cap rows (non-streaming); ignored if split already sliced.")

    p.add_argument("--block-size", type=int, default=512, help="LM sequence length (packed blocks).")
    p.add_argument("--max-steps", type=int, default=500, help="Optimizer steps (not epochs).")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
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
        args.split = args.split if args.split != "train" else "train"
        args.text_column = "text"


def _resolve_torch_dtype(device_type: str, name: str) -> torch.dtype | None:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    # auto
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


def main() -> None:
    args = _parse_args()
    _apply_preset(args)
    if not args.dataset:
        print("Either --preset or --dataset is required.", file=sys.stderr)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model_id} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if args.max_context_len:
        TransformersChatBackend._limit_context_inplace(model, tokenizer=tokenizer, max_context_len=int(args.max_context_len))
    if args.max_layers is not None:
        TransformersChatBackend._truncate_layers_inplace(model, max_layers=int(args.max_layers))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    kwargs_load: Dict[str, Any] = {"split": args.split}
    if args.dataset_config is not None:
        kwargs_load["name"] = args.dataset_config
    raw = load_dataset(args.dataset, **kwargs_load)

    if args.max_samples is not None and "[" not in args.split:
        n = min(len(raw), args.max_samples)
        raw = raw.select(range(n))

    cols = raw.column_names
    if args.text_column not in cols:
        print(f"Column {args.text_column!r} not in {cols}", file=sys.stderr)
        sys.exit(1)

    def tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(batch[args.text_column], add_special_tokens=False)

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=cols,
        desc="Tokenizing",
    )
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)

    block_size = int(args.block_size)
    grouped = tokenized.map(
        lambda batch: _group_texts(batch, block_size),
        batched=True,
        desc=f"Packing blocks ({block_size})",
    )

    if len(grouped) == 0:
        print("No training rows after tokenize/pack. Check --text-column / data.", file=sys.stderr)
        sys.exit(1)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_cuda = device_type == "cuda"
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
        prediction_loss_only=True,
        report_to="none",
        seed=args.seed,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        use_cpu=(device_type == "cpu"),
        use_mps_device=(device_type == "mps"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=grouped,
        data_collator=collator,
    )

    print(f"Train rows (packed blocks): {len(grouped)} | device={device_type} | max_steps={args.max_steps}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
