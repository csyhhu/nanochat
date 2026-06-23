"""
Plot training/eval loss curves from nanochat training logs.
Usage:
    python scripts/plot_training_loss.py logs/qwen6-ft-wiki.log
    python scripts/plot_training_loss.py logs/qwen6-ft-wiki.log --output figures/loss.png
    python scripts/plot_training_loss.py logs/qwen6-ft-wiki.log --no-show  # save only, don't display
    python scripts/plot_training_loss.py logs/qwen6-ft-wiki.log --smooth 0.6  # EMA smoothing

Step determination strategy:
  1. Prefer trainer_state.json (contains precise step numbers in log_history).
  2. Fall back to parsing the text log (train loss by logging index, eval loss
     estimated from the ratio of train/eval entries).
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _detect_encoding(log_path: str) -> str:
    """Detect file encoding from BOM or fallback to utf-8."""
    with open(log_path, "rb") as f:
        bom = f.read(4)
    if bom.startswith(b"\xff\xfe"):
        return "utf-16-le"
    elif bom.startswith(b"\xfe\xff"):
        return "utf-16-be"
    elif bom.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def _find_trainer_state(log_path: str) -> str | None:
    """Look for trainer_state.json near the log file.

    Priority:
      1. output_dir/loss_history.jsonl (custom callback, precise steps, written every logging/eval step)
      2. output_dir/checkpoint-*/trainer_state.json (latest checkpoint, written during training)
      3. output_dir/trainer_state.json (written after training completes)
    """
    log_dir = Path(log_path).parent

    # 1. Check for loss_history.jsonl (custom callback output)
    for p in log_dir.rglob("loss_history.jsonl"):
        return str(p)
    for p in log_dir.parent.rglob("loss_history.jsonl"):
        return str(p)

    # 2-3. Search for checkpoint dirs or trainer_state.json
    checkpoint_states: list[Path] = []
    for p in log_dir.rglob("trainer_state.json"):
        checkpoint_states.append(p)
    for p in log_dir.parent.rglob("trainer_state.json"):
        if p not in checkpoint_states:
            checkpoint_states.append(p)

    if not checkpoint_states:
        return None

    # Return the one with the highest step number (most recent)
    def _extract_step(p: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", str(p))
        return int(match.group(1)) if match else 0

    checkpoint_states.sort(key=_extract_step, reverse=True)
    return str(checkpoint_states[0])


def parse_loss_history(state_path: str) -> tuple[list[dict], list[dict]]:
    """Parse loss_history.jsonl (custom callback output) with precise step numbers.

    Each line is a JSON object like:
      {"step": 10, "train_loss": 5.123, "grad_norm": 1.5}
      {"step": 20, "eval_loss": 4.567}
    """
    train_entries = []
    eval_entries = []
    with open(state_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "train_loss" in entry:
                train_entries.append({"loss": entry["train_loss"], "step": entry["step"]})
            if "eval_loss" in entry:
                eval_entries.append({"eval_loss": entry["eval_loss"], "step": entry["step"]})
    return train_entries, eval_entries


def parse_trainer_state(state_path: str) -> tuple[list[dict], list[dict]]:
    """Parse trainer_state.json to extract train/eval loss with precise step numbers.

    trainer_state.json contains a "log_history" list where each entry has:
      - "step": the global step number
      - "loss": training loss (if it was a training logging step)
      - "eval_loss": evaluation loss (if it was an eval logging step)
    """
    train_entries = []
    eval_entries = []
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log_history = data.get("log_history", [])
    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_entries.append({"loss": entry["loss"], "step": entry["step"]})
        if "eval_loss" in entry and "step" in entry:
            eval_entries.append({"eval_loss": entry["eval_loss"], "step": entry["step"]})
    return train_entries, eval_entries


def parse_log(log_path: str) -> tuple[list[dict], list[dict]]:
    """Parse nanochat training log, extract train_loss and eval_loss entries.

    Returns entries WITHOUT step numbers (step must be inferred later).
    """
    train_entries = []
    eval_entries = []

    encoding = _detect_encoding(log_path)

    with open(log_path, "rb") as f:
        raw = f.read()

    # Strip ANSI escape sequences before decoding
    ansi_escape = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]")
    raw = ansi_escape.sub(b"", raw)
    text = raw.decode(encoding, errors="replace")
    lines = text.splitlines()

    # Detect dict-like lines: starts with {', ends with }
    dict_pattern = re.compile(r"^\{'.+\}$")

    for line in lines:
        line = line.strip()
        if not dict_pattern.match(line):
            continue
        try:
            d = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if "loss" in d and "grad_norm" in d:
            train_entries.append(d)
        elif "eval_loss" in d:
            eval_entries.append(d)

    return train_entries, eval_entries


def ema_smooth(y: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Exponential moving average smoothing."""
    if len(y) == 0:
        return y
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _extract_steps_and_losses(
    train_entries: list[dict],
    eval_entries: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract step numbers and loss values from parsed entries.

    If entries have "step" keys (from trainer_state.json), use them directly.
    Otherwise fall back to index-based numbering.
    """
    # Training loss
    if train_entries and "step" in train_entries[0]:
        train_steps = np.array([e["step"] for e in train_entries], dtype=np.int64)
    else:
        train_steps = np.arange(1, len(train_entries) + 1, dtype=np.int64)
    train_losses = np.array([e["loss"] for e in train_entries], dtype=np.float64)

    # Eval loss
    if eval_entries:
        if "step" in eval_entries[0]:
            eval_steps = np.array([e["step"] for e in eval_entries], dtype=np.int64)
        else:
            # Estimate: assume eval happens at evenly-spaced logging intervals
            eval_interval = max(1, len(train_entries) // max(1, len(eval_entries)))
            eval_steps = np.array(
                [min((i + 1) * eval_interval, train_steps[-1]) for i in range(len(eval_entries))],
                dtype=np.int64,
            )
        eval_losses = np.array([e["eval_loss"] for e in eval_entries], dtype=np.float64)
    else:
        eval_steps = None
        eval_losses = None

    return train_steps, train_losses, eval_steps, eval_losses


def plot_losses(
    train_entries: list[dict],
    eval_entries: list[dict],
    output_path: str | None = None,
    show: bool = True,
    smooth: float | None = None,
):
    """Plot training and eval loss curves."""
    train_steps, train_losses, eval_steps, eval_losses = _extract_steps_and_losses(
        train_entries, eval_entries
    )

    fig, axes = plt.subplots(1, 2 if eval_losses is not None else 1, figsize=(14, 5))
    if eval_losses is None:
        axes = [axes]

    # --- Left: training loss ---
    ax = axes[0]
    ax.plot(train_steps, train_losses, alpha=0.3, color="tab:blue", linewidth=0.8, label="Train Loss (raw)")
    if smooth is not None:
        smoothed = ema_smooth(train_losses, alpha=smooth)
        ax.plot(train_steps, smoothed, color="tab:blue", linewidth=1.5, label=f"Train Loss (EMA α={smooth})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: eval loss ---
    if eval_losses is not None:
        ax = axes[1]
        ax.plot(eval_steps, eval_losses, "o-", color="tab:red", markersize=5, linewidth=1.5, label="Eval Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {len(train_entries)} train points", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training/eval loss from nanochat logs")
    parser.add_argument("log_file", help="Path to the training log file")
    parser.add_argument("--output", "-o", default=None, help="Output image path (png/pdf/svg)")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot")
    parser.add_argument("--smooth", type=float, default=None, help="EMA smoothing factor (0-1, e.g. 0.6)")
    parser.add_argument(
        "--state",
        default=None,
        help="Path to trainer_state.json (auto-detected if not provided)",
    )
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    # --- 1. Try loss_history.jsonl / trainer_state.json first (precise step numbers) ---
    state_path = args.state or _find_trainer_state(args.log_file)
    if state_path:
        if state_path.endswith(".jsonl"):
            print(f"Reading loss_history.jsonl: {state_path}")
            train_entries, eval_entries = parse_loss_history(state_path)
        else:
            print(f"Reading trainer_state.json: {state_path}")
            train_entries, eval_entries = parse_trainer_state(state_path)
        print(f"Found {len(train_entries)} train loss, {len(eval_entries)} eval loss (with precise steps)")
    else:
        print(f"Parsing {args.log_file} ...")
        train_entries, eval_entries = parse_log(args.log_file)
        print(f"Found {len(train_entries)} train loss, {len(eval_entries)} eval loss (estimated steps)")

    if not train_entries:
        print("Error: no training loss entries found.", file=sys.stderr)
        sys.exit(1)

    plot_losses(
        train_entries,
        eval_entries,
        output_path=args.output,
        show=not args.no_show,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
