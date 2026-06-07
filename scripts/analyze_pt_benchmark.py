#!/usr/bin/env python3
"""Reconstruct benchmark rankings from completed run_* folders (no safetensors needed)."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAIN_METRICS_RE = re.compile(
    r"'train_runtime':\s*([\d.]+).*?"
    r"'train_samples_per_second':\s*([\d.]+).*?"
    r"'train_steps_per_second':\s*([\d.]+).*?"
    r"'train_loss':\s*([\d.]+)"
)
TQDM_STEP_RE = re.compile(r"(\d+)/(\d+)\s+\[[\d:]+<[^,]+,\s*([\d.]+)s/it\]")
TRAIN_READY_RE = re.compile(r"Train packed blocks:\s*(\d+)")

GRIDS = {
    "full": {
        "per_device_train_batch_size": [1, 2, 4],
        "gradient_accumulation_steps": [4, 8],
        "block_size": [256, 512],
        "omp_num_threads": [None, 4, 8],
        "gradient_checkpointing": [False, True],
    },
}


def combo_for_index(grid: Dict[str, List[Any]], index: int) -> Dict[str, Any]:
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    return dict(zip(keys, combos[index]))


def parse_log(log_path: Path, max_steps: int) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {
        "completed": "Wrote run summary" in text or "Saved model + tokenizer" in text,
    }
    m = TRAIN_METRICS_RE.search(text)
    if m:
        runtime = float(m.group(1))
        samples_per_sec = float(m.group(2))
        steps_per_sec = float(m.group(3))
        out["train_runtime_sec"] = round(runtime, 1)
        out["samples_per_sec"] = round(samples_per_sec, 4)
        out["steps_per_sec"] = round(steps_per_sec, 4)
        out["sec_per_step"] = round(1.0 / steps_per_sec, 2) if steps_per_sec > 0 else None
        out["sec_per_10_steps"] = round(10.0 / steps_per_sec, 1) if steps_per_sec > 0 else None
        out["final_train_loss"] = float(m.group(4))
    else:
        last_tqdm: Optional[tuple] = None
        for line in text.splitlines():
            tm = TQDM_STEP_RE.search(line)
            if tm:
                cur, total, sec_per_it = int(tm.group(1)), int(tm.group(2)), float(tm.group(3))
                if total == max_steps and cur == max_steps:
                    last_tqdm = (sec_per_it, total)
        if last_tqdm:
            sec_per_it, total = last_tqdm
            out["sec_per_step"] = round(sec_per_it, 2)
            out["sec_per_10_steps"] = round(sec_per_it * 10, 1)
            out["steps_per_sec"] = round(1.0 / sec_per_it, 4)
            out["train_runtime_sec"] = round(sec_per_it * total, 1)
    if TRAIN_READY_RE.search(text):
        out["train_packed_blocks"] = int(TRAIN_READY_RE.search(text).group(1))  # type: ignore[union-attr]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-dir", type=str, default="benchmarks/pt-grid-full")
    p.add_argument("--grid", type=str, default="full")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-rss-frac", type=float, default=0.85)
    args = p.parse_args()

    root = Path(args.benchmark_dir)
    grid = GRIDS[args.grid]

    try:
        import psutil

        host_ram = psutil.virtual_memory().total / (1024**3)
        host_cpus = psutil.cpu_count(logical=True) or 1
    except ImportError:
        host_ram = None
        host_cpus = None

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        idx = int(run_dir.name.split("_")[1])
        combo = combo_for_index(grid, idx)
        log_path = run_dir / "benchmark_stdout.log"
        row: Dict[str, Any] = {"run": run_dir.name, "index": idx, **combo}
        if log_path.exists():
            row.update(parse_log(log_path, args.max_steps))
        summary_path = run_dir / "pt_run_summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            row["final_train_loss"] = s.get("final_train_loss")
            row["train_packed_blocks"] = s.get("train_packed_blocks")
            row["completed"] = True
        rows.append(row)

    ram_limit = host_ram * args.max_rss_frac if host_ram else None
    for r in rows:
        # Primary score: HF train_samples_per_second (fair across batch / grad_accum).
        r["score"] = r.get("samples_per_sec") if r.get("completed") and r.get("samples_per_sec") else None

    completed = [r for r in rows if r.get("completed") and r.get("samples_per_sec")]
    completed.sort(key=lambda r: r["samples_per_sec"], reverse=True)

    out = {
        "host_logical_cpus": host_cpus,
        "host_ram_total_gb": round(host_ram, 2) if host_ram else None,
        "completed_runs_with_timing": len(completed),
        "total_run_dirs": len(rows),
        "top10": completed[:10],
        "best": completed[0] if completed else None,
        "all_runs": rows,
    }

    out_path = root / "benchmark_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("host_logical_cpus", "host_ram_total_gb", "completed_runs_with_timing", "total_run_dirs", "best", "top10")}, indent=2))


if __name__ == "__main__":
    main()
