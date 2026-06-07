#!/usr/bin/env python3
"""
Grid-search short ``qwen_continue_pt`` runs to find a fast CPU training config.

For each parameter combination measures:
  - Host: logical CPU count, total RAM (GiB)
  - Process: peak RSS of the training ``python`` child (GiB)
  - Throughput: wall seconds per 10 optimizer steps (steady train loop only)

Writes ``benchmark_results.json`` / ``benchmark_results.csv`` under ``--output-dir``.
Picks the best combo that maximizes steps/sec while staying under ``--max-rss-frac`` of RAM.

Example (small grid, ~6 runs)::

    cd D:/WorkSpace/nanochat
    $env:PYTHONPATH = (Get-Location).Path
    conda activate nanochat-qwen

    python -m scripts.qwen_pt_benchmark \\
      --output-dir D:/WorkSpace/nanochat/benchmarks/pt-grid \\
      --preset wikitext \\
      --max-layers 6 \\
      --max-samples 800 \\
      --max-steps 30 \\
      --logging-steps 10

Full grid (many runs; use ``--dry-run`` first)::

    python -m scripts.qwen_pt_benchmark --output-dir ./benchmarks/pt-grid --grid full --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

LOSS_RE = re.compile(r"""['"]loss['"]\s*:\s*([0-9.eE+-]+)""")
TRAIN_READY_RE = re.compile(r"Train packed blocks:\s*(\d+)")


@dataclass
class HostInfo:
    logical_cpus: int
    ram_total_gb: float


@dataclass
class RunMetrics:
    success: bool
    error: Optional[str] = None
    train_packed_blocks: Optional[int] = None
    train_wall_sec: Optional[float] = None
    sec_per_10_steps: Optional[float] = None
    steps_per_sec: Optional[float] = None
    proc_rss_gb_max: Optional[float] = None
    cpu_pct_max_core_peak: Optional[float] = None
    loss_samples: List[float] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark qwen_continue_pt hyperparameter grid.")
    p.add_argument("--output-dir", type=str, required=True, help="Results + per-run PT checkpoints parent dir.")
    p.add_argument(
        "--grid",
        type=str,
        default="quick",
        choices=("quick", "full", "minimal"),
        help="Parameter grid preset (see GRIDS in script).",
    )
    p.add_argument("--grid-json", type=str, default=None, help="JSON file overriding --grid, e.g. {\"per_device_train_batch_size\":[1,2]}")
    p.add_argument("--dry-run", action="store_true", help="Print combinations and exit.")
    p.add_argument("--max-runs", type=int, default=None, help="Stop after N combinations (debug).")

    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-layers", type=int, default=6)
    p.add_argument("--device-type", type=str, default="cpu", choices=("cpu", "mps", "cuda"))
    p.add_argument("--preset", type=str, default="wikitext", choices=("wikitext",))
    p.add_argument("--max-samples", type=int, default=800, help="Small train subset for fast benchmark.")
    p.add_argument("--max-steps", type=int, default=30, help="Short train per combo.")
    p.add_argument("--logging-steps", type=int, default=10, help="Also used for sec/10 steps metric.")
    p.add_argument("--warmup-steps", type=int, default=0, help="0 = steadier timing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rss-frac", type=float, default=0.85, help="Combos above this fraction of RAM are disqualified.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable for child runs.")
    return p.parse_args()


GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "minimal": {
        "per_device_train_batch_size": [1, 2],
        "gradient_accumulation_steps": [8],
        "block_size": [512],
        "omp_num_threads": [None],
        "gradient_checkpointing": [False],
    },
    "quick": {
        "per_device_train_batch_size": [1, 2, 4],
        "gradient_accumulation_steps": [8],
        "block_size": [512],
        "omp_num_threads": [None, 8],
        "gradient_checkpointing": [False],
    },
    "full": {
        "per_device_train_batch_size": [1, 2, 4],
        "gradient_accumulation_steps": [4, 8],
        "block_size": [256, 512],
        "omp_num_threads": [None, 4, 8],
        "gradient_checkpointing": [False, True],
    },
}


def _load_grid(args: argparse.Namespace) -> Dict[str, List[Any]]:
    if args.grid_json:
        path = Path(args.grid_json)
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(GRIDS[args.grid])


def _iter_combos(grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _host_info() -> HostInfo:
    import psutil

    return HostInfo(
        logical_cpus=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
    )


def _monitor_process(pid: int, stop: threading.Event, bucket: Dict[str, Any]) -> None:
    import psutil

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    while not stop.is_set():
        try:
            rss_gb = proc.memory_info().rss / (1024**3)
            bucket["proc_rss_gb_max"] = max(bucket.get("proc_rss_gb_max", 0.0), rss_gb)
            per_cpu = psutil.cpu_percent(interval=0.25, percpu=True)
            if per_cpu:
                bucket["cpu_pct_max_core_peak"] = max(
                    bucket.get("cpu_pct_max_core_peak", 0.0),
                    max(per_cpu),
                )
        except psutil.NoSuchProcess:
            break


def _run_one(
    *,
    combo: Dict[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
    host: HostInfo,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        args.python,
        "-m",
        "scripts.qwen_continue_pt",
        "--model-id",
        args.model_id,
        "--device-type",
        args.device_type,
        "--preset",
        args.preset,
        "--max-samples",
        str(args.max_samples),
        "--max-steps",
        str(args.max_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(args.seed),
        "--no-eval",
        "--full-finetune",
        "--benchmark-no-save",
        "--output-dir",
        str(run_dir),
        "--per-device-train-batch-size",
        str(combo["per_device_train_batch_size"]),
        "--gradient-accumulation-steps",
        str(combo["gradient_accumulation_steps"]),
        "--block-size",
        str(combo["block_size"]),
    ]
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])
    if combo.get("gradient_checkpointing"):
        cmd.append("--gradient-checkpointing")

    env = os.environ.copy()
    omp = combo.get("omp_num_threads")
    if omp is not None:
        env["OMP_NUM_THREADS"] = str(int(omp))
        env["MKL_NUM_THREADS"] = str(int(omp))

    metrics = RunMetrics(success=False)
    bucket: Dict[str, Any] = {}
    stop = threading.Event()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    monitor = threading.Thread(target=_monitor_process, args=(proc.pid, stop, bucket), daemon=True)
    monitor.start()

    train_started_at: Optional[float] = None
    first_loss_at: Optional[float] = None
    last_loss_at: Optional[float] = None
    stdout_lines: List[str] = []

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            m_ready = TRAIN_READY_RE.search(line)
            if m_ready:
                metrics.train_packed_blocks = int(m_ready.group(1))
                train_started_at = time.perf_counter()
            m_loss = LOSS_RE.search(line)
            if m_loss:
                metrics.loss_samples.append(float(m_loss.group(1)))
                now = time.perf_counter()
                if first_loss_at is None:
                    first_loss_at = now
                last_loss_at = now
        rc = proc.wait()
    finally:
        stop.set()
        monitor.join(timeout=2.0)

    log_path = run_dir / "benchmark_stdout.log"
    log_path.write_text("".join(stdout_lines), encoding="utf-8")

    metrics.proc_rss_gb_max = bucket.get("proc_rss_gb_max")
    metrics.cpu_pct_max_core_peak = bucket.get("cpu_pct_max_core_peak")

    if rc != 0:
        metrics.error = f"exit code {rc}"
        return _result_row(combo, args, host, metrics)

    metrics.success = True
    if train_started_at is not None:
        end_at = last_loss_at or time.perf_counter()
        metrics.train_wall_sec = end_at - train_started_at
        if metrics.train_wall_sec > 0:
            metrics.steps_per_sec = args.max_steps / metrics.train_wall_sec
            metrics.sec_per_10_steps = metrics.train_wall_sec * 10.0 / args.max_steps

    if first_loss_at is not None and last_loss_at is not None and len(metrics.loss_samples) >= 2:
        span_steps = (len(metrics.loss_samples) - 1) * args.logging_steps
        span_sec = last_loss_at - first_loss_at
        if span_steps > 0 and span_sec > 0:
            metrics.sec_per_10_steps = span_sec * 10.0 / span_steps
            metrics.steps_per_sec = span_steps / span_sec

    return _result_row(combo, args, host, metrics)


def _result_row(
    combo: Dict[str, Any],
    args: argparse.Namespace,
    host: HostInfo,
    metrics: RunMetrics,
) -> Dict[str, Any]:
    ram_limit_gb = host.ram_total_gb * args.max_rss_frac
    rss = metrics.proc_rss_gb_max
    within_ram = rss is not None and rss <= ram_limit_gb
    score = metrics.steps_per_sec if metrics.success and within_ram and metrics.steps_per_sec else None

    return {
        **combo,
        "success": metrics.success,
        "error": metrics.error,
        "host_logical_cpus": host.logical_cpus,
        "host_ram_total_gb": round(host.ram_total_gb, 2),
        "train_packed_blocks": metrics.train_packed_blocks,
        "train_wall_sec": round(metrics.train_wall_sec, 3) if metrics.train_wall_sec is not None else None,
        "sec_per_10_steps": round(metrics.sec_per_10_steps, 3) if metrics.sec_per_10_steps is not None else None,
        "steps_per_sec": round(metrics.steps_per_sec, 4) if metrics.steps_per_sec is not None else None,
        "proc_rss_gb_max": round(metrics.proc_rss_gb_max, 3) if metrics.proc_rss_gb_max is not None else None,
        "cpu_pct_max_core_peak": round(metrics.cpu_pct_max_core_peak, 1)
        if metrics.cpu_pct_max_core_peak is not None
        else None,
        "ram_limit_gb": round(ram_limit_gb, 2),
        "within_ram_limit": within_ram,
        "score_steps_per_sec": round(score, 4) if score is not None else None,
        "effective_batch_size": combo["per_device_train_batch_size"] * combo["gradient_accumulation_steps"],
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
    }


def _pick_best(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    eligible = [r for r in rows if r.get("score_steps_per_sec") is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["score_steps_per_sec"])


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = _parse_args()
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("Install psutil: pip install psutil", file=sys.stderr)
        sys.exit(1)

    grid = _load_grid(args)
    combos = list(_iter_combos(grid))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Grid '{args.grid}' ({len(combos)} combinations)")
    for k, v in grid.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        for i, c in enumerate(combos):
            print(f"  [{i + 1}] {c}")
        return

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    host = _host_info()
    print(
        f"Host: {host.logical_cpus} logical CPUs, {host.ram_total_gb:.1f} GiB RAM | "
        f"RAM cap for scoring: {args.max_rss_frac * 100:.0f}%"
    )

    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(combos):
        run_dir = out_root / f"run_{i:04d}"
        print(f"\n=== Run {i + 1}/{len(combos)} === {combo} ===")
        row = _run_one(combo=combo, run_dir=run_dir, args=args, host=host)
        rows.append(row)
        print(
            f"Result: success={row['success']} steps/s={row.get('steps_per_sec')} "
            f"sec/10steps={row.get('sec_per_10_steps')} RSS_max={row.get('proc_rss_gb_max')} GiB"
        )

    best = _pick_best(rows)
    summary = {
        "grid": grid,
        "host_logical_cpus": host.logical_cpus,
        "host_ram_total_gb": round(host.ram_total_gb, 2),
        "max_rss_frac": args.max_rss_frac,
        "benchmark_args": {
            "model_id": args.model_id,
            "max_layers": args.max_layers,
            "max_samples": args.max_samples,
            "max_steps": args.max_steps,
            "logging_steps": args.logging_steps,
            "device_type": args.device_type,
        },
        "best": best,
        "runs": rows,
    }

    json_path = out_root / "benchmark_results.json"
    csv_path = out_root / "benchmark_results.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    if best:
        print("\nRecommended training flags (copy to qwen_continue_pt):")
        print(
            f"  --per-device-train-batch-size {best['per_device_train_batch_size']} "
            f"--gradient-accumulation-steps {best['gradient_accumulation_steps']} "
            f"--block-size {best['block_size']}"
        )
        if best.get("gradient_checkpointing"):
            print("  --gradient-checkpointing")
        if best.get("omp_num_threads") is not None:
            print(f"  # PowerShell before train: $env:OMP_NUM_THREADS={best['omp_num_threads']}; $env:MKL_NUM_THREADS={best['omp_num_threads']}")
        print(
            f"  # Measured: {best.get('steps_per_sec')} steps/s, "
            f"{best.get('sec_per_10_steps')} s per 10 steps, "
            f"peak RSS {best.get('proc_rss_gb_max')} GiB"
        )
    else:
        print("\nNo successful run within RAM limit. See benchmark_results.json.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
