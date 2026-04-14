"""
wilson_ci.py
------------
Purpose:  Compute 95% Wilson score confidence intervals for each (model, task_type)
          cell and output a LaTeX table matching Appendix C Table 9.
Inputs:   evaluation/results/{model}_results.csv  (one per model)
Outputs:  evaluation/error_analysis/wilson_ci.csv
          LaTeX table printed to stdout
Usage:
    python scripts/wilson_ci.py
"""

import csv
import math
import sys
from pathlib import Path
from typing import NamedTuple

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_DIR = Path("evaluation/results")
OUT_CSV     = Path("evaluation/error_analysis/wilson_ci.csv")

MODEL_ORDER = ["haiku", "gemini", "groq70b", "llama3", "mistral",
               "llama4scout", "qwen3_32b", "deepseek_r1_70b", "gemma4_e4b"]
MODEL_LABELS = {
    "haiku":            "Claude 3 Haiku",
    "gemini":           "Gemini 2.5 Flash",
    "groq70b":          "LLaMA-3.3-70B",
    "llama3":           "LLaMA-3-8B",
    "mistral":          "Mistral-7B",
    "llama4scout":      "Llama 4 Scout 17B",
    "qwen3_32b":        "Qwen3-32B",
    "deepseek_r1_70b":  "DeepSeek R1 70B",
    "gemma4_e4b":       "Gemma 4 E4B",
}

TASK_ORDER = ["regulatory_interpretation", "numerical_reasoning",
              "contradiction_detection",   "temporal_reasoning"]
TASK_SHORT = {"regulatory_interpretation": "REG",
              "numerical_reasoning":       "NUM",
              "contradiction_detection":   "CON",
              "temporal_reasoning":        "TMP"}

# Ground-truth per-task sample sizes (verified against benchmark)
TASK_N: dict[str, int] = {"regulatory_interpretation": 53,
                           "numerical_reasoning":       32,
                           "contradiction_detection":   30,
                           "temporal_reasoning":        35}

# Paper Table 9 reference values (model × task accuracy %) for verification
TABLE9_REF: dict[str, dict[str, float]] = {
    "haiku":   {"REG": 92.5, "NUM": 93.8, "CON": 86.7, "TMP": 91.4},
    "gemini":  {"REG": 96.2, "NUM": 84.4, "CON": 83.3, "TMP": 82.4},
    "groq70b": {"REG": 77.4, "NUM": 84.4, "CON": 90.0, "TMP": 77.1},
    "llama3":  {"REG": 77.4, "NUM": 62.5, "CON": 86.7, "TMP": 74.3},
    "mistral": {"REG": 69.8, "NUM": 68.8, "CON": 80.0, "TMP": 74.3},
}

Z95 = 1.96  # 95% confidence level


# ── Wilson CI formula ──────────────────────────────────────────────────────────

class WilsonCI(NamedTuple):
    n: int
    n_correct: int
    accuracy: float       # point estimate (0–1)
    ci_lower: float       # lower bound (0–1)
    ci_upper: float       # upper bound (0–1)
    center: float         # Wilson centre (0–1)
    half_width: float     # half-width (0–1)


def wilson_ci(n_correct: int, n: int, z: float = Z95) -> WilsonCI:
    """Compute the Wilson score confidence interval.

    Args:
        n_correct: Number of correct predictions.
        n:         Total number of predictions.
        z:         Z-score for desired confidence level (default 1.96 for 95%).

    Returns:
        WilsonCI namedtuple with accuracy, ci_lower, ci_upper, center, half_width.
    """
    if n == 0:
        return WilsonCI(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    p_hat = n_correct / n
    z2n   = z * z / n

    center     = (p_hat + z2n / 2) / (1 + z2n)
    half_width = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / (1 + z2n)

    return WilsonCI(
        n         = n,
        n_correct = n_correct,
        accuracy  = p_hat,
        ci_lower  = max(0.0, center - half_width),
        ci_upper  = min(1.0, center + half_width),
        center    = center,
        half_width= half_width,
    )


# ── Load results ───────────────────────────────────────────────────────────────

def load_model_results(model_key: str) -> dict[str, list[int]]:
    """Load correct/incorrect flags per task for one model.

    Args:
        model_key: Key in MODEL_ORDER (e.g. 'haiku').

    Returns:
        Dict mapping task_type → list of 0/1 correct flags.
    """
    path = RESULTS_DIR / f"{model_key}_results.csv"
    task_scores: dict[str, list[int]] = {t: [] for t in TASK_ORDER}
    if not path.exists():
        print(f"  [WARN] Missing: {path}", file=sys.stderr)
        return task_scores

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "FAIL" in row.get("prediction", ""):
                continue
            t = row.get("task_type", "")
            if t in task_scores:
                task_scores[t].append(int(row.get("correct", 0)))

    return task_scores


# ── Indistinguishability warnings ─────────────────────────────────────────────

def check_indistinguishable(
    cis: dict[str, dict[str, WilsonCI]],
) -> list[str]:
    """Warn if inter-model accuracy difference < sum of half-widths (overlapping CIs).

    Args:
        cis: Nested dict [model_key][task_short] → WilsonCI.

    Returns:
        List of warning strings (empty if all differences are significant).
    """
    warnings: list[str] = []
    for ti, task in enumerate(TASK_ORDER):
        ts = TASK_SHORT[task]
        for i, mk_a in enumerate(MODEL_ORDER):
            for mk_b in MODEL_ORDER[i + 1:]:
                ci_a = cis[mk_a][ts]
                ci_b = cis[mk_b][ts]
                diff = abs(ci_a.accuracy - ci_b.accuracy)
                threshold = ci_a.half_width + ci_b.half_width
                if diff < threshold:
                    warnings.append(
                        f"  [INDISTINGUISHABLE] {MODEL_LABELS[mk_a]} vs "
                        f"{MODEL_LABELS[mk_b]} on {ts}: "
                        f"|{ci_a.accuracy*100:.1f}% - {ci_b.accuracy*100:.1f}%| = "
                        f"{diff*100:.1f}pp < sum-of-half-widths "
                        f"{threshold*100:.1f}pp"
                    )
    return warnings


# ── Table 9 verification ───────────────────────────────────────────────────────

def verify_against_table9(cis: dict[str, dict[str, WilsonCI]]) -> None:
    """Compare computed accuracy against paper Table 9 reference values.

    Prints a discrepancy warning for any cell that differs by > 0.5 pp.

    Args:
        cis: Nested dict [model_key][task_short] → WilsonCI.
    """
    print("\n  Verifying against Table 9 (tolerance ±0.5pp):")
    any_discrepancy = False
    for mk in MODEL_ORDER:
        ref = TABLE9_REF.get(mk, {})
        for ts, ref_acc in ref.items():
            computed = cis[mk][ts].accuracy * 100
            diff = abs(computed - ref_acc)
            if diff > 0.5:
                print(
                    f"  [DISCREPANCY] {MODEL_LABELS[mk]} / {ts}: "
                    f"computed {computed:.1f}% vs paper {ref_acc:.1f}% "
                    f"(Δ={diff:.1f}pp)"
                )
                any_discrepancy = True
    if not any_discrepancy:
        print("  All cells within ±0.5pp of Table 9. OK.")


# ── Output: LaTeX table ────────────────────────────────────────────────────────

def print_latex_table(cis: dict[str, dict[str, WilsonCI]]) -> None:
    """Print a LaTeX table matching Appendix C Table 9 format.

    Args:
        cis: Nested dict [model_key][task_short] → WilsonCI.
    """
    col_spec = "l" + "c" * len(TASK_ORDER)
    header_cols = " & ".join(TASK_SHORT[t] for t in TASK_ORDER)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{95\% Wilson Score Confidence Intervals by Model and Task (Appendix C Table 9)}",
        r"\label{tab:wilson_ci}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Model & {header_cols} \\",
        r"\midrule",
    ]

    for mk in MODEL_ORDER:
        cells = []
        for task in TASK_ORDER:
            ts  = TASK_SHORT[task]
            ci  = cis[mk][ts]
            acc = ci.accuracy * 100
            lo  = ci.ci_lower * 100
            hi  = ci.ci_upper * 100
            cells.append(
                rf"{acc:.1f}\% $[{lo:.1f}, {hi:.1f}]$"
            )
        lines.append(rf"{MODEL_LABELS[mk]} & {' & '.join(cells)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n" + "\n".join(lines))


# ── Output: CSV ────────────────────────────────────────────────────────────────

def save_csv(cis: dict[str, dict[str, WilsonCI]]) -> None:
    """Save Wilson CI results to CSV.

    Args:
        cis: Nested dict [model_key][task_short] → WilsonCI.
    """
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mk in MODEL_ORDER:
        for task in TASK_ORDER:
            ts = TASK_SHORT[task]
            ci = cis[mk][ts]
            rows.append({
                "model":      MODEL_LABELS[mk],
                "task_type":  ts,
                "n":          ci.n,
                "n_correct":  ci.n_correct,
                "accuracy":   round(ci.accuracy, 6),
                "ci_lower":   round(ci.ci_lower, 6),
                "ci_upper":   round(ci.ci_upper, 6),
                "half_width": round(ci.half_width, 6),
            })
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved: {OUT_CSV}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("IndiaFinBench — Wilson Score Confidence Intervals (95%)\n")

    cis: dict[str, dict[str, WilsonCI]] = {}

    for mk in MODEL_ORDER:
        task_scores = load_model_results(mk)
        cis[mk] = {}
        for task in TASK_ORDER:
            ts     = TASK_SHORT[task]
            scores = task_scores[task]
            n_corr = sum(scores)
            n      = len(scores)
            if n == 0:
                n = TASK_N[task]  # fallback to expected size
            cis[mk][ts] = wilson_ci(n_corr, n)

    # Console summary
    header = f"  {'Model':<22} " + "  ".join(f"{TASK_SHORT[t]:>28}" for t in TASK_ORDER)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mk in MODEL_ORDER:
        cells = []
        for task in TASK_ORDER:
            ts = TASK_SHORT[task]
            ci = cis[mk][ts]
            cells.append(
                f"{ci.accuracy*100:5.1f}% [{ci.ci_lower*100:5.1f},{ci.ci_upper*100:5.1f}]"
            )
        print(f"  {MODEL_LABELS[mk]:<22} " + "  ".join(cells))

    # Verify against Table 9
    verify_against_table9(cis)

    # Indistinguishability warnings
    warnings = check_indistinguishable(cis)
    if warnings:
        print("\n  Statistically indistinguishable pairs (CI half-widths overlap):")
        for w in warnings:
            print(w)

    # LaTeX table
    print_latex_table(cis)

    # CSV output
    save_csv(cis)


if __name__ == "__main__":
    main()
