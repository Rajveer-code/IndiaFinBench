"""
update_paper_tables.py
Reads all evaluation CSVs (original 5 + new models), computes per-task accuracy,
Wilson CIs, bootstrap significance, then writes:
  evaluation/error_analysis/combined_leaderboard.csv
  paper/tables/table1_updated.tex         (main results)
  paper/tables/appendix_c_wilson_updated.tex
"""
import csv, json, math, os, re, sys, io, random
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = Path(__file__).parent.parent
RESULTS_DIR = BASE / "evaluation/results"
OUT_LEADERBOARD = BASE / "evaluation/error_analysis/combined_leaderboard.csv"
OUT_TABLE1 = BASE / "paper/tables/table1_updated.tex"
OUT_APPENDIX = BASE / "paper/tables/appendix_c_wilson_updated.tex"

TASK_ABBR = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning": "NUM",
    "contradiction_detection": "CON",
    "temporal_reasoning": "TMP",
}
TASK_N = {"REG": 53, "NUM": 32, "CON": 30, "TMP": 35}

MODEL_LABELS = {
    "haiku_results": "Claude 3 Haiku",
    "gemini_results": "Gemini 2.5 Flash",
    "groq70b_results": "LLaMA-3.3-70B (Groq)",
    "llama3_results": "LLaMA-3-8B",
    "mistral_results": "Mistral-7B",
    "llama4scout_results": "Llama 4 Scout 17B",
    "qwen3_32b_results": "Qwen3-32B",
    "deepseek_r1_70b_results": "DeepSeek R1 70B",
    "gemma4_e4b_results": "Gemma 4 E4B",
    "phi4_results": "Phi-4",
}


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96):
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_p(scores_a, scores_b, n_boot=10000, seed=42):
    """Paired two-sided bootstrap, returns p-value."""
    rng = random.Random(seed)
    obs_delta = sum(scores_a) / len(scores_a) - sum(scores_b) / len(scores_b)
    n = len(scores_a)
    count = 0
    for _ in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        da = sum(scores_a[i] for i in idx) / n
        db = sum(scores_b[i] for i in idx) / n
        if abs(da - db - obs_delta) >= abs(obs_delta):
            count += 1
    return count / n_boot


def load_results(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_model_stats(rows):
    task_scores = defaultdict(list)
    all_scores = []
    for row in rows:
        task = TASK_ABBR.get(row["task_type"], row["task_type"])
        c = int(row["correct"])
        task_scores[task].append(c)
        all_scores.append(c)
    return task_scores, all_scores


def main():
    # Discover all result CSVs
    csv_files = sorted(RESULTS_DIR.glob("*_results.csv"))
    if not csv_files:
        print(f"No result CSVs found in {RESULTS_DIR}")
        return

    print(f"Found {len(csv_files)} result files:")
    for f in csv_files:
        print(f"  {f.name}")

    models_data = {}
    for csv_path in csv_files:
        stem = csv_path.stem  # e.g. "haiku_results"
        label = MODEL_LABELS.get(stem, stem)
        rows = load_results(csv_path)
        if len(rows) < 100:
            print(f"  WARNING: {csv_path.name} only has {len(rows)} rows -- skipping")
            continue
        task_scores, all_scores = compute_model_stats(rows)
        models_data[label] = {
            "task_scores": dict(task_scores),
            "all_scores": all_scores,
            "rows": rows,
            "n_total": len(rows),
        }
        print(f"  Loaded {label}: {len(rows)} rows")

    # Sort by overall accuracy (descending)
    def overall_acc(label):
        s = models_data[label]["all_scores"]
        return sum(s) / len(s) if s else 0
    sorted_models = sorted(models_data.keys(), key=overall_acc, reverse=True)

    # ── Write combined leaderboard CSV ────────────────────────────────────────
    lb_rows = []
    for label in sorted_models:
        md = models_data[label]
        ts = md["task_scores"]
        all_s = md["all_scores"]
        row = {"model": label}
        for task in ["REG", "NUM", "CON", "TMP"]:
            s = ts.get(task, [])
            acc, ci_lo, ci_hi = wilson_ci(sum(s), len(s)) if s else (0, 0, 0)
            row[task] = f"{acc*100:.1f}"
            row[f"{task}_ci_lo"] = f"{ci_lo*100:.1f}"
            row[f"{task}_ci_hi"] = f"{ci_hi*100:.1f}"
        ov_acc, ov_lo, ov_hi = wilson_ci(sum(all_s), len(all_s))
        row["Overall"] = f"{ov_acc*100:.1f}"
        row["Overall_ci_lo"] = f"{ov_lo*100:.1f}"
        row["Overall_ci_hi"] = f"{ov_hi*100:.1f}"
        lb_rows.append(row)

    with open(OUT_LEADERBOARD, "w", newline="", encoding="utf-8") as f:
        fields = ["model","REG","REG_ci_lo","REG_ci_hi","NUM","NUM_ci_lo","NUM_ci_hi",
                  "CON","CON_ci_lo","CON_ci_hi","TMP","TMP_ci_lo","TMP_ci_hi",
                  "Overall","Overall_ci_lo","Overall_ci_hi"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(lb_rows)
    print(f"\nSaved: {OUT_LEADERBOARD}")

    # Print leaderboard
    print("\n=== LEADERBOARD ===")
    header = f"{'Model':<30} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6} {'Overall':>8}"
    print(header)
    print("-" * len(header))
    for row in lb_rows:
        print(f"{row['model']:<30} {row['REG']:>6} {row['NUM']:>6} {row['CON']:>6} {row['TMP']:>6} {row['Overall']:>8}")

    # ── Write LaTeX table1 (main results) ─────────────────────────────────────
    tex_rows = []
    for label in sorted_models:
        md = models_data[label]
        ts = md["task_scores"]
        all_s = md["all_scores"]

        cells = [label]
        for task in ["REG", "NUM", "CON", "TMP"]:
            s = ts.get(task, [])
            if s:
                acc = sum(s) / len(s) * 100
                cells.append(f"{acc:.1f}")
            else:
                cells.append("--")
        ov = sum(all_s) / len(all_s) * 100 if all_s else 0
        cells.append(f"{ov:.1f}")
        tex_rows.append(cells)

    # Bold the best per column
    col_vals = []
    for col_i in range(1, 6):  # REG, NUM, CON, TMP, Overall
        nums = []
        for row in tex_rows:
            try:
                nums.append(float(row[col_i]))
            except ValueError:
                nums.append(0.0)
        col_vals.append(max(nums))

    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Main evaluation results on IndiaFinBench (accuracy \%, 150 items). "
        r"Best per column in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{REG} & \textbf{NUM} & \textbf{CON} & \textbf{TMP} & \textbf{Overall} \\",
        r"\midrule",
    ]
    for row_i, row in enumerate(tex_rows):
        cells_tex = [row[0]]
        for col_i in range(1, 6):
            try:
                val = float(row[col_i])
                cell = f"{val:.1f}"
                if abs(val - col_vals[col_i - 1]) < 0.01:
                    cell = r"\textbf{" + cell + "}"
            except ValueError:
                cell = row[col_i]
            cells_tex.append(cell)
        tex_lines.append(" & ".join(cells_tex) + r" \\")
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUT_TABLE1.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLE1, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Saved: {OUT_TABLE1}")

    # ── Write Wilson CI appendix table ─────────────────────────────────────────
    ci_rows = []
    for label in sorted_models:
        md = models_data[label]
        ts = md["task_scores"]
        all_s = md["all_scores"]
        for task in ["REG", "NUM", "CON", "TMP"]:
            s = ts.get(task, [])
            n = len(s)
            nc = sum(s)
            acc, ci_lo, ci_hi = wilson_ci(nc, n) if n else (0, 0, 0)
            ci_rows.append({
                "model": label, "task": task, "n": n, "n_correct": nc,
                "accuracy": f"{acc*100:.1f}",
                "ci_lower": f"{ci_lo*100:.1f}",
                "ci_upper": f"{ci_hi*100:.1f}",
            })
        n = len(all_s); nc = sum(all_s)
        acc, ci_lo, ci_hi = wilson_ci(nc, n) if n else (0, 0, 0)
        ci_rows.append({
            "model": label, "task": "ALL", "n": n, "n_correct": nc,
            "accuracy": f"{acc*100:.1f}",
            "ci_lower": f"{ci_lo*100:.1f}",
            "ci_upper": f"{ci_hi*100:.1f}",
        })

    ci_tex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{95\% Wilson score confidence intervals for all models.}",
        r"\label{tab:wilson_ci}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Task} & \textbf{n} & \textbf{Acc\%} & \textbf{CI lower} & \textbf{CI upper} \\",
        r"\midrule",
    ]
    prev_model = None
    for row in ci_rows:
        if row["model"] != prev_model and prev_model is not None:
            ci_tex.append(r"\midrule")
        prev_model = row["model"]
        ci_tex.append(
            f"{row['model']} & {row['task']} & {row['n']} & "
            f"{row['accuracy']} & {row['ci_lower']} & {row['ci_upper']}" + r" \\"
        )
    ci_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(OUT_APPENDIX, "w", encoding="utf-8") as f:
        f.write("\n".join(ci_tex) + "\n")
    print(f"Saved: {OUT_APPENDIX}")

    print("\nDone.")


if __name__ == "__main__":
    main()
