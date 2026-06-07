"""Summarize sweep results: mean eval_loss, std, and wall-clock time for each max_samples."""
import os, json, re, collections

eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval_results")
d = collections.defaultdict(list)
times = collections.defaultdict(list)

for f in sorted(os.listdir(eval_dir)):
    if not f.startswith("sweep_n") or not f.endswith(".json"):
        continue
    path = os.path.join(eval_dir, f)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    m = re.search(r'_n(\d+)', f)
    if not m:
        continue
    n = int(m.group(1))
    loss = data.get("eval_loss")
    wall = data.get("wall_clock_sec")
    if loss is None:
        continue
    d[n].append(loss)
    if wall is not None:
        times[n].append(wall)

print(f"{'max_samples':>12}  {'n_trials':>9}  {'avg_eval_loss':>14}  {'std':>10}  {'avg_wall(s)':>12}")
print("-" * 65)
for n in sorted(d.keys()):
    vals = d[n]
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1) if len(vals) > 1 else 0.0
    std = var ** 0.5
    t = times.get(n, [])
    avg_time = sum(t) / len(t) if t else 0.0
    print(f"{n:>12}  {len(vals):>9}  {mean:>14.4f}  {std:>10.4f}  {avg_time:>12.1f}")
