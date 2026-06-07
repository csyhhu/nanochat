#!/usr/bin/env python3
"""
Sweep --eval-max-samples with multiple trials to measure eval_loss variance.

For each --eval-max-samples value, runs N trials with different --eval-seed
values.  This tells you:
1. What is the mean eval_loss at each sample size?
2. What is the within-sample-size variance (std)?
3. At what sample size does the variance become small enough?

Output is a markdown-friendly table and per-trial JSONs under eval_results/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval_results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep eval-max-samples with multiple trials.")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--preset", type=str, default="wikitext")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device-type", type=str, default="cpu")
    p.add_argument(
        "--samples",
        type=str,
        default="10,20,30,50,75,100,150,200,300,500,750,1000",
        help="Comma-separated --eval-max-samples values to test.",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per sample size (different random seeds).",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Starting seed; each trial increments by 1.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trials whose output JSON already exists.",
    )
    return p.parse_args()


def _cleanup_tmp(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def run_one_trial(max_samples: int, seed: int, args: argparse.Namespace) -> Dict[str, Any] | None:
    """Run a single eval with the given max_samples and seed."""
    tag = f"n{max_samples:04d}_s{seed:03d}"
    out_json = RESULTS_DIR / f"sweep_{tag}.json"

    if args.skip_existing and out_json.exists():
        print(f"  [SKIP] {out_json} already exists")
        result = json.loads(out_json.read_text(encoding="utf-8"))
        result["_tag"] = tag
        return result

    cmd = [
        sys.executable, "-m", "scripts.qwen_continue_pt",
        "--eval-only",
        "--model-id", args.model_id,
        "--max-layers", str(args.max_layers),
        "--preset", args.preset,
        "--block-size", str(args.block_size),
        "--per-device-eval-batch-size", str(args.batch_size),
        "--device-type", args.device_type,
        "--eval-max-samples", str(max_samples),
        "--eval-seed", str(seed),
        "--output-json", str(out_json),
    ]

    print(f"  [RUN] max_samples={max_samples:>5}  seed={seed:>3}  -> {tag}")
    t0 = time.perf_counter()

    env = os.environ.copy()
    env.pop("HF_ENDPOINT", None)

    # Write stdout/stderr to temp files to avoid pipe buffer issues on Windows.
    tmp_stdout = out_json.with_suffix(".stdout.tmp")
    tmp_stderr = out_json.with_suffix(".stderr.tmp")
    try:
        with open(tmp_stdout, "w", encoding="utf-8") as fout, \
             open(tmp_stderr, "w", encoding="utf-8") as ferr:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=fout,
                stderr=ferr,
                env=env,
                timeout=3600,
            )
    except subprocess.TimeoutExpired:
        print(f"         [TIMEOUT] >1h")
        _cleanup_tmp(tmp_stdout, tmp_stderr)
        return None

    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"         [FAILED] rc={proc.returncode}")
        try:
            stderr_text = tmp_stderr.read_text(encoding="utf-8", errors="replace")
        except Exception:
            stderr_text = ""
        for line in stderr_text.strip().splitlines()[-5:]:
            print(f"         stderr: {line}")
        _cleanup_tmp(tmp_stdout, tmp_stderr)
        return None

    _cleanup_tmp(tmp_stdout, tmp_stderr)

    if not out_json.exists():
        print(f"         [FAILED] JSON not found")
        return None

    result = json.loads(out_json.read_text(encoding="utf-8"))
    result["_tag"] = tag
    eval_loss = result.get("eval_loss", float("nan"))
    n_blocks = result.get("eval_blocks", 0)
    print(f"         loss={eval_loss:.6f}  blocks={n_blocks}  wall={elapsed:.1f}s")
    return result


def _fmt_num(v: float, decimals: int = 4) -> str:
    if v is None or math.isnan(v):
        return " " * (decimals + 2 + 1)  # space for sign
    return f"{v:+.{decimals}f}"


def main() -> None:
    args = parse_args()
    samples_list = [int(x.strip()) for x in args.samples.split(",") if x.strip()]
    num_trials = args.num_trials
    base_seed = args.base_seed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_id}  layers={args.max_layers}  preset={args.preset}")
    print(f"Block size: {args.block_size}  batch: {args.batch_size}  device: {args.device_type}")
    print(f"Sweep values: {samples_list}")
    print(f"Trials per size: {num_trials}  (seeds {base_seed}..{base_seed + num_trials - 1})")
    print(f"Total runs: {len(samples_list) * num_trials}")
    print()

    # Collect all trial results, grouped by max_samples
    grouped: Dict[int, List[Dict[str, Any]]] = {}

    for n in samples_list:
        grouped[n] = []
        for trial_idx in range(num_trials):
            seed = base_seed + trial_idx
            r = run_one_trial(n, seed, args)
            if r is not None:
                grouped[n].append(r)

    # ------------------------------------------------------------------
    # Summary: mean / std / min / max per sample size
    # ------------------------------------------------------------------
    print(f"\n\n{'='*95}")
    print("PER-SIZE SUMMARY (eval_loss statistics across trials)")
    print(f"{'='*95}")
    header = (
        f"{'max_s':>6}  {'trials':>6}  {'mean_loss':>10}  {'std':>8}  "
        f"{'min':>10}  {'max':>10}  {'range':>8}  {'blocks':>7}  {'wall_avg':>8}"
    )
    print(header)
    print("-" * 95)

    stats_rows: List[Dict[str, Any]] = []
    for n in samples_list:
        trials = grouped.get(n, [])
        losses = [t["eval_loss"] for t in trials if t.get("eval_loss") is not None]
        blocks_list = [t.get("eval_blocks", 0) for t in trials]
        wall_list = [t.get("wall_clock_sec", 0) for t in trials]

        if len(losses) < 2:
            print(f"{n:>6}  {len(losses):>6}  {'(need >=2)':>10}")
            stats_rows.append({"max_samples": n, "count": len(losses)})
            continue

        mean_loss = statistics.mean(losses)
        std_loss = statistics.stdev(losses)
        min_loss = min(losses)
        max_loss = max(losses)
        avg_blocks = statistics.mean(blocks_list) if blocks_list else 0
        avg_wall = statistics.mean(wall_list) if wall_list else 0

        print(
            f"{n:>6}  {len(losses):>6}  {mean_loss:>10.6f}  {std_loss:>8.6f}  "
            f"{min_loss:>10.6f}  {max_loss:>10.6f}  {max_loss-min_loss:>8.6f}  "
            f"{avg_blocks:>7.0f}  {avg_wall:>8.1f}"
        )
        stats_rows.append({
            "max_samples": n,
            "count": len(losses),
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "min_loss": min_loss,
            "max_loss": max_loss,
            "range": max_loss - min_loss,
            "avg_blocks": avg_blocks,
            "avg_wall": avg_wall,
        })

    print("-" * 95)

    # ------------------------------------------------------------------
    # Per-trial detail table
    # ------------------------------------------------------------------
    print(f"\n{'='*95}")
    print("PER-TRIAL DETAIL")
    print(f"{'='*95}")
    detail_header = f"{'max_s':>6}  {'seed':>5}  {'loss':>10}  {'blocks':>7}  {'wall':>8}"
    print(detail_header)
    print("-" * 45)
    for n in samples_list:
        for t in grouped.get(n, []):
            loss = t.get("eval_loss", float("nan"))
            blk = t.get("eval_blocks", 0)
            wall = t.get("wall_clock_sec", 0)
            seed = t.get("_tag", "").split("_s")[-1] if "_s" in t.get("_tag", "") else "?"
            print(f"{n:>6}  {seed:>5}  {loss:>10.6f}  {blk:>7}  {wall:>8.1f}")
    print("-" * 45)

    # ------------------------------------------------------------------
    # Variance trend: how std evolves with sample size
    # ------------------------------------------------------------------
    print(f"\n{'='*95}")
    print("VARIANCE TREND (std vs max_samples)")
    print(f"{'='*95}")
    print(f"{'max_s':>6}  {'std_loss':>10}  {'coeff_var':>10}  {'avg_blocks':>10}  {'1/sqrt(blocks)':>14}")
    print("-" * 60)
    for row in stats_rows:
        if "std_loss" not in row:
            continue
        cv = row["std_loss"] / row["mean_loss"] * 100 if row["mean_loss"] != 0 else 0
        blk = row["avg_blocks"]
        inv_sqrt = 1.0 / math.sqrt(blk) if blk > 0 else 0
        print(
            f"{row['max_samples']:>6}  {row['std_loss']:>10.6f}  {cv:>9.2f}%  "
            f"{blk:>10.0f}  {inv_sqrt:>14.6f}"
        )
    print("-" * 60)
    print("  coeff_var = std / mean * 100%  (lower is better)")
    print("  1/sqrt(blocks): expected scaling of std if blocks were i.i.d.")

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    valid_stats = [r for r in stats_rows if "mean_loss" in r]
    if len(valid_stats) >= 2:
        print(f"\n--- Recommendation ---")
        for cv_threshold in [0.5, 1.0, 2.0, 3.0]:
            ref = valid_stats[-1]
            good = [
                r for r in valid_stats
                if r.get("std_loss", 999) / r.get("mean_loss", 1) * 100 < cv_threshold
            ]
            if good:
                best = min(good, key=lambda r: r["max_samples"])
                print(
                    f"  CV < {cv_threshold:>4.1f}%:  max_samples >= {best['max_samples']}  "
                    f"(mean_loss={best['mean_loss']:.6f}, std={best['std_loss']:.6f}, "
                    f"avg_blocks={best['avg_blocks']:.0f}, "
                    f"avg_wall={best['avg_wall']:.1f}s)"
                )
                break

    print(f"\nAll trial JSONs saved under: {RESULTS_DIR}/sweep_n*_s*.json")
    print("Done.")


if __name__ == "__main__":
    main()
