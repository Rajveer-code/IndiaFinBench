"""
bootstrap_significance.py
--------------------------
Purpose:  Paired bootstrap significance tests for all 10 model pairs on
          IndiaFinBench.  Also runs per-task tests for the 3 most interesting
          comparisons (Haiku vs Gemini, Haiku vs LLaMA-70B, Gemini vs LLaMA-70B).
Inputs:   evaluation/results/{model}_results.csv  (one per model)
Outputs:  evaluation/error_analysis/bootstrap_significance.csv
          Formatted console report
Usage:
    python scripts/bootstrap_significance.py
"""

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_DIR = Path("evaluation/results")
OUT_CSV     = Path("evaluation/error_analysis/bootstrap_significance.csv")

MODEL_ORDER = ["haiku", "gemini", "groq70b", "llama3", "mistral"]
MODEL_LABELS = {
    "haiku":   "Claude 3 Haiku",
    "gemini":  "Gemini 2.5 Flash",
    "groq70b": "LLaMA-3.3-70B",
    "llama3":  "LLaMA-3-8B",
    "mistral": "Mistral-7B",
}

TASK_ORDER = ["regulatory_interpretation", "numerical_reasoning",
              "contradiction_detection",   "temporal_reasoning"]
TASK_SHORT = {"regulatory_interpretation": "REG",
              "numerical_reasoning":       "NUM",
              "contradiction_detection":   "CON",
              "temporal_reasoning":        "TMP"}

# Top 3 comparisons for per-task breakdown
FOCUS_PAIRS = [
    ("haiku",   "gemini"),
    ("haiku",   "groq70b"),
    ("gemini",  "groq70b"),
]

N_RESAMPLES = 10_000
ALPHA       = 0.05
SEED        = 42

np.random.seed(SEED)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_aligned_scores(
    mk_a: str, mk_b: str
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load per-item correct/incorrect scores for two models, aligned by item ID.

    Items must be scored on the same set of QA items.  We intersect IDs so that
    the bootstrap operates on matched pairs.

    Args:
        mk_a: First model key.
        mk_b: Second model key.

    Returns:
        Tuple of:
          - scores_a: Overall aligned correct array for model A.
          - scores_b: Overall aligned correct array for model B.
          - per_task:  Dict task_short -> (scores_a_task, scores_b_task).
    """
    def read_csv(mk: str) -> dict[str, dict]:
        path = RESULTS_DIR / f"{mk}_results.csv"
        if not path.exists():
            print(f"  [WARN] Missing: {path}", file=sys.stderr)
            return {}
        rows = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    rows[row["id"]] = row
        return rows

    rows_a = read_csv(mk_a)
    rows_b = read_csv(mk_b)
    shared = sorted(set(rows_a) & set(rows_b))

    scores_a = np.array([int(rows_a[iid]["correct"]) for iid in shared])
    scores_b = np.array([int(rows_b[iid]["correct"]) for iid in shared])

    per_task: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in TASK_ORDER:
        ts    = TASK_SHORT[task]
        idxs  = [i for i, iid in enumerate(shared)
                 if rows_a[iid].get("task_type", "") == task]
        if idxs:
            per_task[ts] = (scores_a[idxs], scores_b[idxs])

    return scores_a, scores_b, per_task


# ── Bootstrap test ─────────────────────────────────────────────────────────────

def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = N_RESAMPLES,
) -> float:
    """Two-sided paired bootstrap test for accuracy difference.

    Null hypothesis: mean(A) = mean(B).
    Uses the "shift" method: centre each distribution at zero, then count how
    often the resampled difference exceeds the observed difference.

    Args:
        scores_a:    Binary correct array for model A (aligned with B).
        scores_b:    Binary correct array for model B (aligned with B).
        n_resamples: Number of bootstrap resamples.

    Returns:
        Two-sided p-value.
    """
    n = len(scores_a)
    if n == 0:
        return 1.0

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    # Shift each series to null (zero mean difference)
    delta   = scores_a - scores_b
    centred = delta - np.mean(delta)

    rng   = np.random.default_rng(SEED)
    idx   = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = centred[idx].mean(axis=1)

    # Two-sided: proportion of resamples where |diff| >= |observed|
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))
    return float(p_value)


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(
    results: list[dict],
    per_task_results: list[dict],
) -> None:
    """Print a formatted significance report to stdout.

    Args:
        results:          List of overall comparison result dicts.
        per_task_results: List of per-task comparison result dicts.
    """
    sep = "-" * 80
    print(f"\n{'='*80}")
    print("  IndiaFinBench — Paired Bootstrap Significance Tests")
    print(f"  n_resamples={N_RESAMPLES:,}  alpha={ALPHA}  seed={SEED}")
    print(f"{'='*80}\n")

    print(f"  {'Model A':<22}  {'Model B':<22}  {'Acc A':>6}  {'Acc B':>6}  {'Delta':>6}  {'p-val':>7}  Sig?")
    print(f"  {sep}")
    for r in results:
        sig = "YES *" if r["p_value"] < ALPHA else "no"
        print(
            f"  {r['model_a']:<22}  {r['model_b']:<22}  "
            f"{r['acc_a']*100:>5.1f}%  {r['acc_b']*100:>5.1f}%  "
            f"{r['delta']*100:>+5.1f}%  {r['p_value']:>7.4f}  {sig}"
        )

    if per_task_results:
        print(f"\n  Per-task breakdown (top 3 comparisons):")
        print(f"  {sep}")
        print(f"  {'Comparison':<46}  {'Task':>4}  {'Acc A':>6}  {'Acc B':>6}  {'p-val':>7}  Sig?")
        print(f"  {sep}")
        for r in per_task_results:
            sig  = "YES *" if r["p_value"] < ALPHA else "no"
            pair = f"{r['model_a']} vs {r['model_b']}"
            print(
                f"  {pair:<46}  {r['task']:>4}  "
                f"{r['acc_a']*100:>5.1f}%  {r['acc_b']*100:>5.1f}%  "
                f"{r['p_value']:>7.4f}  {sig}"
            )

    n_sig = sum(1 for r in results if r["p_value"] < ALPHA)
    print(f"\n  {n_sig}/{len(results)} overall comparisons significant at alpha={ALPHA}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all pairwise bootstrap tests and save results."""
    all_pairs  = list(combinations(MODEL_ORDER, 2))  # 10 pairs
    results:          list[dict] = []
    per_task_results: list[dict] = []

    print("  Running bootstrap tests...", file=sys.stderr)

    for mk_a, mk_b in all_pairs:
        scores_a, scores_b, per_task = load_aligned_scores(mk_a, mk_b)
        if len(scores_a) == 0:
            continue

        p_val = paired_bootstrap(scores_a, scores_b)
        results.append({
            "model_a":  MODEL_LABELS[mk_a],
            "model_b":  MODEL_LABELS[mk_b],
            "n":        len(scores_a),
            "acc_a":    float(np.mean(scores_a)),
            "acc_b":    float(np.mean(scores_b)),
            "delta":    float(np.mean(scores_a) - np.mean(scores_b)),
            "p_value":  p_val,
            "significant": p_val < ALPHA,
        })

        # Per-task tests for focus pairs only
        if (mk_a, mk_b) in FOCUS_PAIRS:
            for ts, (sa, sb) in per_task.items():
                pt_p = paired_bootstrap(sa, sb)
                per_task_results.append({
                    "model_a":     MODEL_LABELS[mk_a],
                    "model_b":     MODEL_LABELS[mk_b],
                    "task":        ts,
                    "n":           len(sa),
                    "acc_a":       float(np.mean(sa)),
                    "acc_b":       float(np.mean(sb)),
                    "p_value":     pt_p,
                    "significant": pt_p < ALPHA,
                })

    # Console report
    print_report(results, per_task_results)

    # Save CSV — unify schema by adding a 'task' column (overall = 'ALL')
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    unified: list[dict] = []
    for r in results:
        unified.append({**r, "task": "ALL"})
    for r in per_task_results:
        # per_task rows already have 'task'; add missing overall-only fields
        unified.append({**r, "n": r["n"], "delta": r["acc_a"] - r["acc_b"]})
    if unified:
        fields = ["model_a", "model_b", "task", "n", "acc_a", "acc_b",
                  "delta", "p_value", "significant"]
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(unified)
        print(f"  Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
