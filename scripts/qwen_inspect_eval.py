"""
Inspect Qwen model outputs on eval datasets.

Shows exactly what the model sees (prompt) and what it outputs (logits/generation)
for each task type. Useful for debugging why eval scores are low and understanding
how the model behaves on different task formats.

Usage:
    # Inspect MMLU (categorical task) with sft mode, first 5 examples
    python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --tasks MMLU --num-examples 5

    # Inspect all tasks, 3 examples each
    python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --num-examples 3

    # Inspect base model with base mode
    python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B --mode base --tasks MMLU --num-examples 3

    # Inspect with 6-layer truncation
    python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --max-layers 6 --tasks MMLU,GSM8K --num-examples 3

    # Inspect generative task with sampling
    python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --tasks GSM8K --num-examples 2 --temperature 0.8
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from nanochat.transformers_backend import (
    TransformersChatBackend,
    _prefer_offline_hub_load,
    resolve_hf_model_path,
)

# -----------------------------------------------------------------------------
# Adapters (re-use from qwen_eval.py)
# -----------------------------------------------------------------------------

class QwenModelAdapter:
    def __init__(self, model: Any):
        self.model = model

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(input_ids)
        return output.logits

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device


class QwenTokenizerAdapter:
    def __init__(self, tokenizer: Any, mode: str = "sft"):
        self.tokenizer = tokenizer
        self.mode = mode

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def get_bos_token_id(self) -> int:
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            return bos
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            return eos
        return 0

    def render_for_completion(self, conversation: Dict[str, Any]) -> List[int]:
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()

        if self.mode == "base":
            return self._render_base(conv)
        else:
            return self._render_chat(messages)

    def _render_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist()
            return ids

        text_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render_base(self, conv: Dict[str, Any]) -> List[int]:
        messages = conv["messages"]
        letters = conv.get("letters", None)
        text_parts = []
        for m in messages:
            content = m.get("content", "")
            if m["role"] == "user":
                text_parts.append(content)
            elif m["role"] == "assistant":
                text_parts.append(content)
        text = "\n".join(text_parts)
        if letters:
            text += "\nAnswer: "
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids


# -----------------------------------------------------------------------------
# Task loading
# -----------------------------------------------------------------------------

ALL_TASKS = ["MMLU", "ARC-Easy", "ARC-Challenge", "GSM8K", "HumanEval", "SpellingBee"]


def _create_task(task_name: str) -> Any:
    from tasks.mmlu import MMLU
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K
    from tasks.humaneval import HumanEval
    from tasks.spellingbee import SpellingBee

    task_map = {
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "HumanEval": partial(HumanEval),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }
    return task_map[task_name]()


# -----------------------------------------------------------------------------
# Inspection functions
# -----------------------------------------------------------------------------

def inspect_categorical(
    task_name: str,
    task_object: Any,
    tokenizer_adapter: QwenTokenizerAdapter,
    model_adapter: QwenModelAdapter,
    num_examples: int,
    device: torch.device,
):
    """Inspect a categorical task (MMLU, ARC-Easy, ARC-Challenge)."""
    print(f"\n{'=' * 80}")
    print(f"TASK: {task_name} (categorical)")
    print(f"{'=' * 80}")

    for i in range(min(num_examples, len(task_object))):
        conversation = task_object[i]
        letters = conversation.get("letters", [])
        ground_truth = conversation["messages"][-1]["content"]  # e.g. "A"

        # Render the prompt
        prompt_ids = tokenizer_adapter.render_for_completion(conversation)
        prompt_text = tokenizer_adapter.decode(prompt_ids)

        print(f"\n{'─' * 80}")
        print(f"Example {i + 1} | Ground truth: {ground_truth} | Letters: {letters}")
        print(f"{'─' * 80}")

        # Show the full prompt
        print(f"\n[FULL PROMPT]:")
        print(prompt_text)
        print(f"\n[PROMPT] token count: {len(prompt_ids)}")

        # Generate: let the model complete the prompt (greedy)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = input_ids.shape[-1]
        with torch.no_grad():
            output = model_adapter.model.generate(
                input_ids,
                max_new_tokens=32,       # categorical answers are short
                do_sample=False,          # greedy
                pad_token_id=tokenizer_adapter.tokenizer.pad_token_id or tokenizer_adapter.tokenizer.eos_token_id,
                eos_token_id=tokenizer_adapter.tokenizer.eos_token_id,
            )
        generated_ids = output[0, prompt_len:].tolist()
        generated_text = tokenizer_adapter.decode(generated_ids)

        print(f"\n[MODEL OUTPUT]:")
        print(generated_text)

        # Result: check if the first non-whitespace char matches ground truth
        first_char = generated_text.strip()[:1] if generated_text.strip() else ""
        correct = first_char == ground_truth
        print(f"\n[RESULT] First char: '{first_char}' | Ground truth: '{ground_truth}' | {'✓ CORRECT' if correct else '✗ WRONG'}")


def inspect_generative(
    task_name: str,
    task_object: Any,
    tokenizer_adapter: QwenTokenizerAdapter,
    model: Any,
    tokenizer: Any,
    num_examples: int,
    device: torch.device,
    temperature: float,
    max_new_tokens: int,
    top_k: int,
):
    """Inspect a generative task (GSM8K, HumanEval, SpellingBee)."""
    print(f"\n{'=' * 80}")
    print(f"TASK: {task_name} (generative)")
    print(f"{'=' * 80}")

    for i in range(min(num_examples, len(task_object))):
        conversation = task_object[i]

        # Get ground truth info
        assistant_msg = conversation["messages"][-1]
        if isinstance(assistant_msg["content"], list):
            # Multi-part content (GSM8K, SpellingBee)
            gt_parts = assistant_msg["content"]
            gt_text = ""
            for part in gt_parts:
                if part["type"] == "text":
                    gt_text += part["text"]
                elif part["type"] == "python":
                    gt_text += f"<<{part['text']}>>"
                elif part["type"] == "python_output":
                    gt_text += f"<<{part['text']}>>"
        else:
            gt_text = str(assistant_msg["content"])

        # Render the prompt
        prompt_ids = tokenizer_adapter.render_for_completion(conversation)
        prompt_text = tokenizer_adapter.decode(prompt_ids)

        print(f"\n{'─' * 80}")
        print(f"Example {i + 1}")
        print(f"{'─' * 80}")

        # Show the prompt
        print(f"\n[PROMPT]:")
        print(prompt_text)
        print(f"\n[PROMPT] token count: {len(prompt_ids)}")

        # Show full ground truth
        print(f"\n[FULL GROUND TRUTH]:")
        print(gt_text)

        # Generate
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = input_ids.shape[-1]

        do_sample = temperature > 0.0
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else None,
                pad_token_id=tokenizer_adapter.tokenizer.pad_token_id or tokenizer_adapter.tokenizer.eos_token_id,
                eos_token_id=tokenizer_adapter.tokenizer.eos_token_id,
            )

        generated_ids = output[0, prompt_len:].tolist()
        generated_text = tokenizer_adapter.decode(generated_ids)

        print(f"\n[GENERATED] ({len(generated_ids)} tokens):")
        print(generated_text)

        # Evaluate
        try:
            outcome = task_object.evaluate(conversation, generated_text)
            print(f"\n[RESULT] {'✓ CORRECT' if outcome else '✗ WRONG'}")
        except Exception as e:
            print(f"\n[RESULT] Evaluation error: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect Qwen model outputs on eval datasets."
    )
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", type=str, default="sft", choices=("base", "sft", "rl"))
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--max-context-len", type=int, default=2048)
    p.add_argument("--torch-dtype", type=str, default="auto",
                   choices=("auto", "float32", "bfloat16", "float16"))
    p.add_argument("--device-type", type=str, default="cpu",
                   choices=("cpu", "mps", "cuda"))
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated task names. Default: all 6 tasks.")
    p.add_argument("--num-examples", type=int, default=3,
                   help="Number of examples to inspect per task.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for generative tasks (0 = greedy).")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max new tokens for generative tasks.")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-k for generative sampling.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve tasks
    if args.tasks is None:
        task_names = list(ALL_TASKS)
    else:
        task_names = [t.strip() for t in args.tasks.replace("|", ",").split(",") if t.strip()]

    for t in task_names:
        if t not in ALL_TASKS:
            print(f"Unknown task: {t}. Available: {ALL_TASKS}", file=sys.stderr)
            sys.exit(1)

    # Set HF mirror
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # Device
    device_type = args.device_type
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    torch_dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = torch_dtype_map.get(args.torch_dtype)
    if args.torch_dtype == "auto" and device_type == "mps":
        torch_dtype = torch.float16

    # Load model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"Missing transformers: {e}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_hf_model_path(args.model_id)
    model_path, local_files_only = _prefer_offline_hub_load(args.model_id, model_path)
    pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)

    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **pretrained_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_path}")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **pretrained_kw
    )
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    if args.max_context_len:
        TransformersChatBackend._limit_context_inplace(
            model, tokenizer=tokenizer, max_context_len=int(args.max_context_len)
        )
    if args.max_layers is not None:
        TransformersChatBackend._truncate_layers_inplace(model, max_layers=int(args.max_layers))
        print(f"Truncated to {args.max_layers} layers")

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.1f}M params on {device}")

    # Build adapters
    model_adapter = QwenModelAdapter(model)
    tokenizer_adapter = QwenTokenizerAdapter(tokenizer, mode=args.mode)

    print(f"\n{'#' * 80}")
    print(f"Inspect: mode={args.mode}, tasks={task_names}, examples={args.num_examples}")
    print(f"  model_id={args.model_id}, max_layers={args.max_layers}")
    print(f"{'#' * 80}")

    # Inspect each task
    for task_name in task_names:
        task_object = _create_task(task_name)
        eval_type = task_object.eval_type

        if eval_type == "categorical":
            inspect_categorical(
                task_name, task_object, tokenizer_adapter, model_adapter,
                num_examples=args.num_examples, device=device,
            )
        elif eval_type == "generative":
            inspect_generative(
                task_name, task_object, tokenizer_adapter, model, tokenizer,
                num_examples=args.num_examples, device=device,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
            )
        else:
            print(f"  Unknown eval_type: {eval_type}")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
