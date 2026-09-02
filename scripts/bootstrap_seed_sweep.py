"""Plan v3 Phase 3.3: run scripts/provenance_and_clustered_bootstrap.py under
several seeds at 100,000 resamples and check whether the reported significant-
pair / Bonferroni-survivor counts are seed-stable.

The paper's own record noted the Bonferroni-survivor count moved between 15
and 16 across independent 10,000-resample runs. That is not acceptable in a
paper whose thesis is about measurement precision. This script is the check:
if every seed agrees, report the stable count. If not, report the range and
say so in the manuscript instead of a point estimate.

Does not touch evaluation/clustered_bootstrap.json or clustered_bootstrap_full66.json
(the canonical, seed=42 outputs) -- each sweep run writes to a seed-suffixed
file via BOOTSTRAP_OUT_SUFFIX so the canonical files are never clobbered by a
sweep run that isn't the final one.

Usage: python scripts/bootstrap_seed_sweep.py [seed1 seed2 ...]
Default seeds: 42 (canonical), 7, 123, 2026, 99999
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 7, 123, 2026, 99999]
ROOT = Path(__file__).parent.parent

summaries = []
for seed in SEEDS:
    suffix = "" if seed == 42 else f"_seed{seed}"
    env = os.environ.copy()
    env["BOOTSTRAP_N_RESAMPLES"] = "100000"
    env["BOOTSTRAP_SEED"] = str(seed)
    env["BOOTSTRAP_OUT_SUFFIX"] = suffix
    print(f"\n{'='*70}\nseed={seed} (writing evaluation/clustered_bootstrap_full66{suffix}.json)\n{'='*70}")
    r = subprocess.run([sys.executable, "scripts/provenance_and_clustered_bootstrap.py"],
                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"seed={seed} failed, returncode={r.returncode}")
    out_path = ROOT / "evaluation" / f"clustered_bootstrap_full66{suffix}.json"
    d = json.loads(out_path.read_text(encoding="utf-8"))
    summaries.append({
        "seed": seed,
        "n_sig_item_05": d["n_sig_item_05"], "n_sig_cluster_05": d["n_sig_cluster_05"],
        "n_sig_item_bonferroni": d["n_sig_item_bonferroni"],
        "n_sig_cluster_bonferroni": d["n_sig_cluster_bonferroni"],
        "n_pairs_flip_direction_05": d["n_pairs_flip_direction_05"],
    })

print(f"\n{'='*70}\nSTABILITY SUMMARY across {len(SEEDS)} seeds\n{'='*70}")
fields = ["n_sig_item_05", "n_sig_cluster_05", "n_sig_item_bonferroni",
          "n_sig_cluster_bonferroni", "n_pairs_flip_direction_05"]
print(f"{'seed':>8s} " + " ".join(f"{f:>26s}" for f in fields))
for s in summaries:
    print(f"{s['seed']:>8d} " + " ".join(f"{s[f]:>26d}" for f in fields))

stable = {}
for f in fields:
    vals = {s[f] for s in summaries}
    stable[f] = (len(vals) == 1, sorted(vals))
    tag = "STABLE" if stable[f][0] else f"UNSTABLE range={stable[f][1]}"
    print(f"  {f}: {tag}")

all_stable = all(v[0] for v in stable.values())
print(f"\n{'ALL METRICS SEED-STABLE -- report the point count.' if all_stable else 'NOT ALL STABLE -- report a range in the manuscript, not a point count.'}")

with open(ROOT / "evaluation" / "bootstrap_stability_sweep.json", "w", encoding="utf-8") as f:
    json.dump({"seeds": SEEDS, "n_resamples": 100000, "runs": summaries,
                "stability": {k: {"stable": v[0], "values": v[1]} for k, v in stable.items()},
                "all_stable": all_stable}, f, indent=2)
print("Saved evaluation/bootstrap_stability_sweep.json")
