#!/usr/bin/env python3
"""
Parse nanochat base_train tee logs and plot train loss + validation bpb.

Usage (from repo root):
  python dev/plot_pretrain_log.py pretrain_mac_baseline.log
  python dev/plot_pretrain_log.py /path/to/log.txt -o loss.png

Requires matplotlib (uv: `uv sync --group dev` or pip install matplotlib).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_LOSS_RE = re.compile(
    r"step\s+(\d+)/\d+\s+\([^)]+\)\s+\|\s+loss:\s*([\d.]+)",
    re.IGNORECASE,
)
VAL_BPB_RE = re.compile(
    r"Step\s+(\d+)\s+\|\s+Validation\s+bpb:\s*([\d.]+)",
    re.IGNORECASE,
)


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_log(text: str) -> tuple[list[int], list[float], list[int], list[float]]:
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_bpbs: list[float] = []

    for raw in text.splitlines():
        line = strip_ansi(raw)
        m = STEP_LOSS_RE.search(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            continue
        m = VAL_BPB_RE.search(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_bpbs.append(float(m.group(2)))

    return train_steps, train_losses, val_steps, val_bpbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot base_train loss from a log file.")
    parser.add_argument(
        "log_file",
        type=Path,
        help="Path to tee log (e.g. pretrain_mac_baseline.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <log_stem>_loss.png next to log)",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: uv sync --group dev\n"
            "or: pip install matplotlib"
        ) from e

    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    steps, losses, val_steps, val_bpbs = parse_log(text)

    if not steps and not val_steps:
        raise SystemExit(
            f"No training or validation lines found in {args.log_file}. "
            "Expected patterns like 'step 00001/02000 ... | loss: 10.39' and "
            "'Step 00050 | Validation bpb: 3.24'."
        )

    out = args.output
    if out is None:
        out = args.log_file.with_name(f"{args.log_file.stem}_loss.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1])
    ax0, ax1 = axes

    if steps:
        ax0.plot(steps, losses, color="C0", linewidth=0.8, label="train loss (debiased EMA)")
        ax0.set_ylabel("train loss")
        ax0.legend(loc="upper right")
        ax0.grid(True, alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "no train loss lines", ha="center", va="center", transform=ax0.transAxes)

    if val_steps:
        ax1.plot(val_steps, val_bpbs, color="C1", marker="o", markersize=3, linewidth=1, label="val bpb")
        ax1.set_ylabel("val bpb")
        ax1.set_xlabel("step")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "no validation lines", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("step")

    fig.suptitle(f"base_train: {args.log_file.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
