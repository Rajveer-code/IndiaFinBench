"""
generate_paper_tables.py
------------------------
Purpose:  Generate all LaTeX tables for the IndiaFinBench paper.
Inputs:   evaluation/results/{model}_results.csv  (one per model)
Outputs:  paper/tables/table5_main_results.tex
          paper/tables/table6_difficulty.tex
          paper/tables/table7_error_distribution.tex
          paper/tables/table9_wilson_ci.tex
Usage:
    python scripts/generate_paper_tables.py
"""

import csv
import math
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

RESULTS_DIR = Path("evaluation/results")
OUT_DIR     = Path("paper/tables")

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

DIFFICULTIES = ["easy", "medium", "hard"]

# Error counts from paper Table 7 (ground truth — do not recompute from CSVs)
ERROR_DATA: dict[str, dict[str, int]] = {
    "haiku":   {"DKF": 4,  "NRF": 2,  "TRF": 7,  "CGF": 0},
    "gemini":  {"DKF": 3,  "NRF": 5,  "TRF": 9,  "CGF": 1},
    "groq70b": {"DKF": 12, "NRF": 5,  "TRF": 10, "CGF": 1},
    "llama3":  {"DKF": 13, "NRF": 12, "TRF": 11, "CGF": 1},
    "mistral": {"DKF": 17, "NRF": 10, "TRF": 13, "CGF": 1},
}
ERROR_TYPE_LABELS = {
    "DKF": "Domain Knowledge",
    "NRF": "Numerical Reasoning",
    "TRF": "Temporal Reasoning",
    "CGF": "Context Grounding",
}

# Wilson CI constants
Z95 = 1.96
TASK_N: dict[str, int] = {"REG": 53, "NUM": 32, "CON": 30, "TMP": 35}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_results() -> dict[str, list[dict]]:
    """Load all model result CSVs.

    Returns:
        Dict mapping model_key -> list of valid row dicts.
    """
    all_rows: dict[str, list[dict]] = {}
    for mk in MODEL_ORDER:
        path = RESULTS_DIR / f"{mk}_results.csv"
        if not path.exists():
            print(f"  [WARN] Missing: {path}", file=sys.stderr)
            all_rows[mk] = []
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    rows.append(row)
        all_rows[mk] = rows
    return all_rows


# ── Accuracy helpers ───────────────────────────────────────────────────────────

def task_acc(rows: list[dict], task: str) -> tuple[float, int]:
    """Compute accuracy for a specific task.

    Args:
        rows: List of result row dicts.
        task: Full task_type string.

    Returns:
        (accuracy_pct, n_items) tuple.
    """
    task_rows = [r for r in rows if r["task_type"] == task]
    if not task_rows:
        return 0.0, 0
    n_corr = sum(int(r["correct"]) for r in task_rows)
    return n_corr / len(task_rows) * 100, len(task_rows)


def diff_acc(rows: list[dict], difficulty: str) -> float:
    """Compute accuracy for a specific difficulty level.

    Args:
        rows:       List of result row dicts.
        difficulty: One of 'easy', 'medium', 'hard'.

    Returns:
        Accuracy as percentage.
    """
    d_rows = [r for r in rows if r.get("difficulty", "") == difficulty]
    if not d_rows:
        return 0.0
    return sum(int(r["correct"]) for r in d_rows) / len(d_rows) * 100


def overall_acc(rows: list[dict]) -> float:
    """Compute overall accuracy across all tasks.

    Args:
        rows: List of result row dicts.

    Returns:
        Accuracy as percentage.
    """
    if not rows:
        return 0.0
    return sum(int(r["correct"]) for r in rows) / len(rows) * 100


def bold_best(values: list[float], fmt: str = "{:.1f}") -> list[str]:
    """Bold the maximum value in a list for LaTeX output.

    Args:
        values: List of float values.
        fmt:    Format string for the number.

    Returns:
        List of LaTeX-formatted strings, with best value bolded.
    """
    best = max(values)
    result = []
    for v in values:
        s = fmt.format(v)
        result.append(rf"\textbf{{{s}}}" if v == best else s)
    return result


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(n_correct: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    """Compute Wilson score 95% CI.

    Args:
        n_correct: Number of correct predictions.
        n:         Total number of predictions.
        z:         Z-score (default 1.96 for 95%).

    Returns:
        (ci_lower_pct, ci_upper_pct, half_width_pct) as percentages.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat  = n_correct / n
    z2n    = z * z / n
    center = (p_hat + z2n / 2) / (1 + z2n)
    hw     = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))) / (1 + z2n)
    return max(0.0, center - hw) * 100, min(1.0, center + hw) * 100, hw * 100


# ── Table 5: Main Results ──────────────────────────────────────────────────────

def make_table5(all_rows: dict[str, list[dict]]) -> str:
    """Generate Table 5: accuracy by task type.

    Args:
        all_rows: Dict model_key -> list of result row dicts.

    Returns:
        LaTeX table string.
    """
    # Collect per-column values to find best per column
    col_vals: dict[str, list[float]] = {ts: [] for ts in list(TASK_SHORT.values()) + ["Overall"]}
    for mk in MODEL_ORDER:
        rows = all_rows.get(mk, [])
        for task in TASK_ORDER:
            ts  = TASK_SHORT[task]
            acc, _ = task_acc(rows, task)
            col_vals[ts].append(acc)
        col_vals["Overall"].append(overall_acc(rows))

    task_headers = " & ".join(TASK_SHORT[t] for t in TASK_ORDER)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{IndiaFinBench: Main Results --- Accuracy (\%) by Task Type (Table 5)}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        rf"Model & {task_headers} & Overall \\",
        r"\midrule",
    ]

    for col in col_vals:
        col_vals[col] = bold_best(col_vals[col])  # type: ignore[assignment]

    col_idx = {ts: 0 for ts in col_vals}
    for mk in MODEL_ORDER:
        cells = []
        for task in TASK_ORDER:
            ts = TASK_SHORT[task]
            cells.append(col_vals[ts][col_idx[ts]])
            col_idx[ts] += 1
        cells.append(col_vals["Overall"][col_idx["Overall"]])
        col_idx["Overall"] += 1
        lines.append(rf"{MODEL_LABELS[mk]} & {' & '.join(cells)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table 6: Difficulty Breakdown ─────────────────────────────────────────────

def make_table6(all_rows: dict[str, list[dict]]) -> str:
    """Generate Table 6: accuracy by difficulty level.

    Args:
        all_rows: Dict model_key -> list of result row dicts.

    Returns:
        LaTeX table string.
    """
    diff_headers = " & ".join(d.capitalize() for d in DIFFICULTIES)

    # Find best per column
    col_vals: dict[str, list[float]] = {d: [] for d in DIFFICULTIES}
    for mk in MODEL_ORDER:
        rows = all_rows.get(mk, [])
        for d in DIFFICULTIES:
            col_vals[d].append(diff_acc(rows, d))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{IndiaFinBench: Accuracy (\%) by Question Difficulty (Table 6)}",
        r"\label{tab:difficulty}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Model & {diff_headers} \\",
        r"\midrule",
    ]

    col_bold: dict[str, list[str]] = {d: bold_best(col_vals[d]) for d in DIFFICULTIES}
    for i, mk in enumerate(MODEL_ORDER):
        cells = [col_bold[d][i] for d in DIFFICULTIES]
        lines.append(rf"{MODEL_LABELS[mk]} & {' & '.join(cells)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table 7: Error Distribution ───────────────────────────────────────────────

def make_table7() -> str:
    """Generate Table 7: error type distribution using hardcoded error_data.

    Returns:
        LaTeX table string.
    """
    et_order = ["DKF", "NRF", "TRF", "CGF"]
    et_header = " & ".join(
        rf"\textbf{{{ERROR_TYPE_LABELS[et]}}}" for et in et_order
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{IndiaFinBench: Error Type Distribution by Model (Table 7)}",
        r"\label{tab:error_distribution}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Model & {et_header} \\",
        r"\midrule",
    ]

    for mk in MODEL_ORDER:
        ed     = ERROR_DATA[mk]
        total  = sum(ed.values()) or 1
        cells  = []
        for et in et_order:
            cnt = ed[et]
            pct = cnt / total * 100
            cells.append(rf"{cnt} ({pct:.0f}\%)")
        lines.append(rf"{MODEL_LABELS[mk]} & {' & '.join(cells)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Table 9: Wilson CIs ───────────────────────────────────────────────────────

def make_table9(all_rows: dict[str, list[dict]]) -> str:
    """Generate Table 9: Wilson score confidence intervals.

    Uses the same formula as wilson_ci.py to avoid duplication.

    Args:
        all_rows: Dict model_key -> list of result row dicts.

    Returns:
        LaTeX table string.
    """
    task_headers = " & ".join(TASK_SHORT[t] for t in TASK_ORDER)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{95\% Wilson Score Confidence Intervals by Model and Task (Table 9)}",
        r"\label{tab:wilson_ci}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Model & {task_headers} \\",
        r"\midrule",
    ]

    for mk in MODEL_ORDER:
        rows = all_rows.get(mk, [])
        cells = []
        for task in TASK_ORDER:
            ts     = TASK_SHORT[task]
            t_rows = [r for r in rows if r["task_type"] == task]
            n_corr = sum(int(r["correct"]) for r in t_rows)
            n      = len(t_rows) or TASK_N[ts]
            acc    = (n_corr / n * 100) if n else 0.0
            lo, hi, _ = wilson_ci(n_corr, n)
            cells.append(rf"{acc:.1f}\% $[{lo:.1f},\,{hi:.1f}]$")
        lines.append(rf"{MODEL_LABELS[mk]} & {' & '.join(cells)} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all 4 paper tables and save to paper/tables/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = load_all_results()

    tables = {
        "table5_main_results.tex":     make_table5(all_rows),
        "table6_difficulty.tex":       make_table6(all_rows),
        "table7_error_distribution.tex": make_table7(),
        "table9_wilson_ci.tex":        make_table9(all_rows),
    }

    for filename, content in tables.items():
        path = OUT_DIR / filename
        path.write_text(content + "\n", encoding="utf-8")
        print(f"  Saved: {path}")

    print("\n  All 4 tables generated.")


if __name__ == "__main__":
    main()
