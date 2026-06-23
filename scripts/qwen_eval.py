"""
Qwen model evaluation script.
Evaluates a Qwen model (base/instruct/RL-tuned) on the 6 standard nanochat
Chat benchmark tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SpellingBee.

Supports:
- Layer truncation (--max-layers)
- Three modes: base, sft, rl (controls how conversations are rendered)
- Categorical tasks (no sampling needed, fast on CPU)
- Generative tasks (sampling, slower)
- Optional max-problems cap for quick smoke tests

Examples::

    # Eval base model on MMLU (6 layers, first 100 problems)
    python -m scripts.qwen_eval --model-id Qwen/Qwen2.5-0.5B --mode base --max-layers 6 --tasks MMLU --max-problems 100

    # Eval instruct model on all 6 tasks (6 layers)
    python -m scripts.qwen_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --max-layers 6

    # Eval RL model on GSM8K only
    python -m scripts.qwen_eval --model-id ./output/qwen-rl --mode rl --max-layers 6 --tasks GSM8K

    # Eval multiple tasks with | separator
    python -m scripts.qwen_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --max-layers 6 --tasks "MMLU|ARC-Easy"
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
# QwenModelAdapter: wraps HF Qwen model for nanochat-compatible interface
# -----------------------------------------------------------------------------

class QwenModelAdapter:
    """
    Lightweight wrapper to give Qwen HF models a nanochat-compatible interface.

    Required by:
    - run_categorical_eval: needs model(input_ids) -> logits, model.get_device()
    - run_generative_eval: needs model.get_device() (generation is done via engine)
    """

    def __init__(self, model: Any, max_seq_len: Optional[int] = None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits. (B, T) -> (B, T, V)"""
        with torch.no_grad():
            output = self.model(input_ids)
        return output.logits

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device


# -----------------------------------------------------------------------------
# QwenTokenizerAdapter: wraps HF Qwen tokenizer for nanochat eval interface
# -----------------------------------------------------------------------------

class QwenTokenizerAdapter:
    """
    Wraps a Qwen HF tokenizer to provide the interface expected by
    run_categorical_eval / run_generative_eval in chat_eval.py:

    - encode(text) -> List[int]
    - decode(ids) -> str
    - get_bos_token_id() -> int
    - render_for_completion(conversation) -> List[int]
    """

    def __init__(self, tokenizer: Any, mode: str = "sft"):
        """
        Args:
            tokenizer: HF AutoTokenizer for Qwen.
            mode: 'base', 'sft', or 'rl'.
                  - 'sft'/'rl': use Qwen chat_template (add_generation_prompt=True)
                  - 'base': use simple text concatenation (base model doesn't understand chat format)
        """
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
        # fallback: use eos as pad/BOS-like token
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            return eos
        return 0

    def render_for_completion(self, conversation: Dict[str, Any]) -> List[int]:
        """
        Render a conversation (from tasks/) into token ids, priming the
        model for an assistant completion.

        The conversation dict has:
        - "messages": list of {"role": "user"/"assistant", "content": str}
        - "letters": (optional) available answer letters for categorical tasks

        We remove the last assistant message (which contains the ground truth)
        and render the rest with the appropriate format.
        """
        # Deep copy to avoid mutating the original
        conv = copy.deepcopy(conversation)
        messages = conv["messages"]

        # The last message should be from the assistant (ground truth)
        # Remove it so the model generates the answer itself
        if messages and messages[-1]["role"] == "assistant":
            messages.pop()

        if self.mode == "base":
            # Base model: simple text concatenation (no chat template)
            # This is a best-effort ICL-style prompt
            return self._render_base(conv)
        else:
            # SFT/RL model: use Qwen's native chat template
            return self._render_chat(messages)

    def _render_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        """Render using Qwen's chat_template with add_generation_prompt=True."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # appends <|im_start|>assistant\n
            )
            # apply_chat_template may return a tensor or list
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist()
            return ids

        # Fallback: naive concatenation
        text_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text_parts.append("<|im_start|>assistant\n")
        text = "".join(text_parts)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render_base(self, conv: Dict[str, Any]) -> List[int]:
        """
        Render for base model: simple ICL-style format.
        Base models don't understand chat templates, so we format
        as a natural text prompt.
        """
        messages = conv["messages"]
        letters = conv.get("letters", None)

        # Build a simple prompt from the user message(s)
        text_parts = []
        for m in messages:
            content = m.get("content", "")
            if m["role"] == "user":
                text_parts.append(content)
            elif m["role"] == "assistant":
                text_parts.append(content)

        text = "\n".join(text_parts)

        # For categorical tasks with multiple choice, hint the model
        if letters:
            text += "\nAnswer: "

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids


# -----------------------------------------------------------------------------
# QwenEngine: minimal generation wrapper for generative tasks
# -----------------------------------------------------------------------------

class QwenEngine:
    """
    Minimal engine for batched generation, compatible with
    the interface expected by run_generative_eval:

        engine.generate_batch(prompt_ids, num_samples, max_tokens, temperature, top_k)
            -> (list of token sequences, None)
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: List[int],
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 50,
    ) -> Tuple[List[List[int]], Any]:
        """
        Generate num_samples completions for the given prompt.
        Returns (list of token sequences, None).
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
        )

        # For num_samples > 1 with do_sample=True, we call generate multiple times
        # (a bit slow but works on CPU)
        all_results = []
        for _ in range(num_samples):
            gen = self.model.generate(input_ids=input_ids, **gen_kwargs)  # type: ignore
            all_results.append(gen[0].tolist())  # strip batch dim

        return all_results, None


# -----------------------------------------------------------------------------
# Categorical evaluation (adapted from chat_eval.py)
# -----------------------------------------------------------------------------

def run_categorical_eval(
    task_object: Any,
    tokenizer: QwenTokenizerAdapter,
    model: QwenModelAdapter,
    batch_size: int = 4,
    max_problems: Optional[int] = None,
) -> float:
    """
    Evaluate a categorical (multiple-choice) task.
    No sampling needed — just compare logits on answer letters.

    Adapted from chat_eval.py to work with Qwen model/tokenizer adapters.
    """
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache: Dict[str, int] = {}
    num_passed, total = 0, 0

    t0 = time.perf_counter()
    for bi in range(num_batches):
        i0, i1 = bi * batch_size, min((bi + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]

        # Render each conversation to token ids and collect answer-time positions
        prompt_ids_list: List[List[int]] = []
        answer_time_positions: List[int] = []
        for conv in conversations:
            ids = tokenizer.render_for_completion(conv)
            prompt_ids_list.append(ids)
            answer_time_positions.append(len(ids) - 1)

        # Pad to max length
        max_length = max(len(ids) for ids in prompt_ids_list)
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids_list]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Forward pass — get logits for the whole batch
        logits = model(input_ids)  # (B, T, V)

        # For each problem, check the logits at the answer position for the
        # available letter tokens
        for idx, conversation in enumerate(conversations):
            letters = conversation.get("letters", [])
            if not letters:
                continue  # skip if no letters defined (shouldn't happen for categorical)

            # Get token ids for each letter
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded = tokenizer.encode(letter)
                    # Each letter should ideally be a single token
                    letter_to_id_cache[letter] = encoded[0] if encoded else 0
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]

            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

        # Progress
        elapsed = time.perf_counter() - t0
        if bi % 10 == 0 or bi == num_batches - 1:
            rate = total / elapsed if elapsed > 0 else 0
            print(f"\r[{bi + 1}/{num_batches} batches] {num_passed}/{total} ({100*num_passed/total:.1f}%) | "
                  f"{rate:.1f} samples/s", end="", flush=True)

    print()  # newline after progress
    acc = num_passed / total if total > 0 else 0.0
    elapsed_total = time.perf_counter() - t0
    print(f"Categorical eval finished: {num_passed}/{total} ({100*acc:.2f}%) in {elapsed_total:.1f}s")
    return acc


# -----------------------------------------------------------------------------
# Generative evaluation (adapted from chat_eval.py)
# -----------------------------------------------------------------------------

def run_generative_eval(
    task_object: Any,
    tokenizer: QwenTokenizerAdapter,
    model: QwenModelAdapter,
    engine: QwenEngine,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: Optional[int] = None,
) -> float:
    """
    Evaluate a generative (open-ended) task by sampling completions.

    Adapted from chat_eval.py to work with Qwen model/tokenizer adapters.
    """
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0

    t0 = time.perf_counter()
    for i in range(num_problems):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(rt[prefix_length:]) for rt in results]
        outcomes = [task_object.evaluate(conversation, c) for c in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)

        elapsed = time.perf_counter() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"\r[{i + 1}/{num_problems}] {num_passed}/{total} ({100*num_passed/total:.1f}%) | "
              f"{rate:.1f} samples/s", end="", flush=True)

    print()  # newline after progress
    acc = num_passed / total if total > 0 else 0.0
    elapsed_total = time.perf_counter() - t0
    print(f"Generative eval finished: {num_passed}/{total} ({100*acc:.2f}%) in {elapsed_total:.1f}s")
    return acc


# -----------------------------------------------------------------------------
# Task registry and orchestration
# -----------------------------------------------------------------------------

ALL_TASKS = ["MMLU", "ARC-Easy", "ARC-Challenge", "GSM8K", "HumanEval", "SpellingBee"]
BASELINE_ACCURACIES = {
    "MMLU": 0.25,
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}


def _create_task(task_name: str, max_problems: Optional[int] = None) -> Any:
    """Create a task object by name, matching chat_eval.py conventions."""
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
        "HumanEval": partial(HumanEval, split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}. Available: {ALL_TASKS}")
    return task_map[task_name]()


def run_qwen_eval(
    task_names: List[str],
    model: Any,
    tokenizer: Any,
    mode: str = "sft",
    batch_size: int = 4,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: Optional[int] = None,
) -> Dict[str, float]:
    """
    Run evaluation on the given tasks for a Qwen model.

    Returns a dict mapping task_name -> accuracy.
    """
    # Build adapters
    model_adapter = QwenModelAdapter(model)
    tokenizer_adapter = QwenTokenizerAdapter(tokenizer, mode=mode)
    engine = QwenEngine(model, tokenizer)

    results: Dict[str, float] = {}
    for task_name in task_names:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name}")
        print(f"{'=' * 60}")

        task_object = _create_task(task_name)
        eval_type = task_object.eval_type

        if eval_type == "categorical":
            acc = run_categorical_eval(
                task_object, tokenizer_adapter, model_adapter,
                batch_size=batch_size, max_problems=max_problems,
            )
        elif eval_type == "generative":
            acc = run_generative_eval(
                task_object, tokenizer_adapter, model_adapter, engine,
                num_samples=num_samples, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, max_problems=max_problems,
            )
        else:
            raise ValueError(f"Unsupported eval_type: {eval_type}")

        results[task_name] = acc
        print(f"  {task_name} accuracy: {100 * acc:.2f}%")

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Qwen models on nanochat Chat benchmark tasks."
    )
    p.add_argument(
        "--model-id", type=str, default="Qwen/Qwen2.5-0.5B",
        help="HF model id or local path.",
    )
    p.add_argument(
        "--mode", type=str, default="sft",
        choices=("base", "sft", "rl"),
        help="Model type: base (no chat template), sft (instruct/chat), rl (RL-tuned).",
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
        help="Device for evaluation.",
    )
    p.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task names, e.g. 'MMLU,ARC-Easy'. Default: all 6 tasks.",
    )
    p.add_argument(
        "--max-problems", type=int, default=None,
        help="Cap number of problems per task (for quick smoke tests).",
    )
    p.add_argument(
        "-t", "--temperature", type=float, default=0.0,
        help="Sampling temperature for generative tasks.",
    )
    p.add_argument(
        "-m", "--max-new-tokens", type=int, default=512,
        help="Max new tokens for generative tasks.",
    )
    p.add_argument(
        "-n", "--num-samples", type=int, default=1,
        help="Number of samples for pass@k in generative tasks.",
    )
    p.add_argument(
        "-k", "--top-k", type=int, default=50,
        help="Top-k sampling for generative tasks.",
    )
    p.add_argument(
        "-b", "--batch-size", type=int, default=4,
        help="Batch size for categorical evaluation.",
    )
    p.add_argument(
        "--output-json", type=str, default=None,
        help="If set, write results as JSON to this path.",
    )
    return p.parse_args()


def _resolve_torch_dtype(device_type: str, name: str) -> Optional[Any]:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if device_type == "mps":
        return torch.float16
    return None  # auto: let HF decide (usually float32)


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

    # Set HF mirror if not already set
    if not os.environ.get("HF_ENDPOINT", "").strip():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # Determine device
    device_type = args.device_type
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    torch_dtype = _resolve_torch_dtype(device_type, args.torch_dtype)

    # Load model and tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as e:
        print(
            "Missing dependency. Install with:\n"
            "  uv sync --group dev\n"
            "or: pip install transformers\n\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = resolve_hf_model_path(args.model_id)
    model_path, local_files_only = _prefer_offline_hub_load(args.model_id, model_path)
    pretrained_kw = dict(trust_remote_code=False, local_files_only=local_files_only)

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **pretrained_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {model_path}" + (" (local cache only)" if local_files_only else ""))
    t_load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **pretrained_kw,
    )
    print(f"Model loaded in {time.perf_counter() - t_load_start:.1f}s")

    # Apply truncations
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

    # Run evaluation
    print(f"\n{'#' * 60}")
    print(f"Qwen Eval: mode={args.mode}, tasks={task_names}")
    print(f"  model_id={args.model_id}")
    print(f"  max_layers={args.max_layers}")
    print(f"  max_problems={args.max_problems}")
    print(f"{'#' * 60}")

    results = run_qwen_eval(
        task_names=task_names,
        model=model,
        tokenizer=tokenizer,
        mode=args.mode,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        max_problems=args.max_problems,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Task':<20} {'Accuracy':>10} {'Baseline':>10} {'Centered':>10}")
    print(f"{'-' * 50}")
    centered_sum = 0.0
    for task_name in ALL_TASKS:
        if task_name in results:
            acc = results[task_name]
            baseline = BASELINE_ACCURACIES.get(task_name, 0.0)
            centered = (acc - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0
            centered_sum += centered
            print(f"{task_name:<20} {100*acc:>9.2f}% {100*baseline:>9.1f}% {centered:>10.4f}")

    if results:
        chatcore = centered_sum / len(ALL_TASKS) if set(ALL_TASKS).issubset(results.keys()) else None
        if chatcore is not None:
            print(f"\nChatCORE: {chatcore:.4f}")
        else:
            # Compute partial ChatCORE on available tasks
            partial_chatcore = centered_sum / len(results)
            print(f"\nPartial ChatCORE ({len(results)}/{len(ALL_TASKS)} tasks): {partial_chatcore:.4f}")

    # Write output JSON if requested
    if args.output_json:
        import json as json_module
        from pathlib import Path
        out = {
            "model_id": args.model_id,
            "mode": args.mode,
            "max_layers": args.max_layers,
            "device_type": device_type,
            "results": {k: round(v, 6) for k, v in results.items()},
        }
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_module.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
