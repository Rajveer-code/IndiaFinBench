"""Rebuild Appendix C1 (CON balanced accuracy) and the error-taxonomy table
from live, corrected data (post Gemma-fix, post phi4-mini full-coverage judge).

Replaces hand-maintained numbers in _appendix_raw.tex / _error_analysis_raw.tex
that were computed before either fix landed. Uses the same deterministic
error-type mapping already described in the manuscript text (task_type x
difficulty -> error type), reused verbatim from scripts/error_analysis.py's
ERROR_TAXONOMY dict rather than redefined here.

Outputs:
  evaluation/con_balance_recomputed.json
  evaluation/error_taxonomy_recomputed.json
  paper/tables/table_con_balance.tex
  paper/tables/table_errortax.tex
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "evaluation" / "results"
JUDGED_PHI4 = ROOT / "evaluation" / "results_judged_phi4"

# Same 12 models, display order matches every other table in the paper.
MODEL_FILES = {
    "Gemini 2.5 Flash": "gemini_results.csv",
    "Qwen3-32B": "qwen3_32b_results.csv",
    "LLaMA-3.3-70B": "groq70b_results.csv",
    "Llama 4 Scout 17B": "llama4scout_results.csv",
    "Kimi K2": "kimi_k2_results.csv",
    "LLaMA-3-8B": "llama3_results.csv",
    "GPT-OSS 120B": "gpt_oss_120b_results.csv",
    "GPT-OSS 20B": "gpt_oss_20b_results.csv",
    "Gemini 2.5 Pro": "gemini25_pro_results.csv",
    "Mistral-7B": "mistral_results.csv",
    "DeepSeek-R1-Distill-Llama-70B": "deepseek_r1_70b_results.csv",
    "Gemma 4 E4B": "gemma4_e4b_results.csv",
}

# Errortax table: the 5 models the manuscript profiles (top / bottom / three
# distinctive profiles) -- unchanged selection from the existing table.
ERRORTAX_MODELS = [
    ("Gemini 2.5 Flash", "gemini_results.csv"),
    ("Qwen3-32B", "qwen3_32b_results.csv"),
    ("LLaMA-3.3-70B", "groq70b_results.csv"),
    ("DeepSeek-R1-Distill-Llama-70B", "deepseek_r1_70b_results.csv"),
    ("Gemma 4 E4B", "gemma4_e4b_results.csv"),
]

# Reused verbatim from scripts/error_analysis.py -- the deterministic mapping
# already described in the manuscript text. Do not redefine independently.
ERROR_TAXONOMY = {
    "regulatory_interpretation": {"easy": "DKF", "medium": "DKF", "hard": "CGF"},
    "numerical_reasoning": {"easy": "NRF", "medium": "NRF", "hard": "NRF"},
    "contradiction_detection": {"easy": "CGF", "medium": "TRF", "hard": "TRF"},
    "temporal_reasoning": {"easy": "TRF", "medium": "TRF", "hard": "DKF"},
}


def load_dataset_difficulty():
    ds_path = ROOT / "annotation" / "raw_qa" / "indiafinbench_qa_combined_406.json"
    data = json.loads(ds_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", data.get("data", list(data.values())[0]))
    return {item["id"]: item["difficulty"] for item in data}


def load_results(fname):
    with open(RESULTS / fname, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Part 1: CON balanced accuracy (Table C1) ─────────────────────────────────

def con_balanced_accuracy():
    out = {}
    yes_total = no_total = None
    for label, fname in MODEL_FILES.items():
        rows = load_results(fname)
        con = [r for r in rows if r["task_type"] == "contradiction_detection"]
        yes_items = [r for r in con if r["ref_answer"].strip().lower() == "yes"]
        no_items = [r for r in con if r["ref_answer"].strip().lower() == "no"]
        yes_total, no_total = len(yes_items), len(no_items)
        tp = sum(1 for r in yes_items if int(r["correct"]) == 1)
        tn = sum(1 for r in no_items if int(r["correct"]) == 1)
        sens = tp / len(yes_items) if yes_items else 0.0
        spec = tn / len(no_items) if no_items else 0.0
        acc = sum(int(r["correct"]) for r in con) / len(con) * 100
        out[label] = {
            "con_acc": round(acc, 1),
            "sensitivity": round(sens * 100, 1),
            "specificity": round(spec * 100, 1),
            "balanced_acc": round((sens + spec) / 2 * 100, 1),
        }
    total = yes_total + no_total
    majority_baseline = max(yes_total, no_total) / total * 100
    return out, yes_total, no_total, majority_baseline


# ── Part 2: Error taxonomy, strict vs judge-confirmed residual ──────────────

def error_taxonomy():
    diff_map = load_dataset_difficulty()
    out = {}
    for label, results_fname in ERRORTAX_MODELS:
        strict_rows = {r["id"]: r for r in load_results(results_fname)}
        # results_judged_phi4/ files are named identically to results/ files
        judged_rows = {r["id"]: r for r in csv.DictReader(
            open(JUDGED_PHI4 / results_fname, newline="", encoding="utf-8"))}

        strict_counts = {"DKF": 0, "NRF": 0, "TRF": 0, "CGF": 0}
        residual_counts = {"DKF": 0, "NRF": 0, "TRF": 0, "CGF": 0}

        for iid, row in strict_rows.items():
            if int(row["correct"]) == 1:
                continue  # not a strict error
            task = row["task_type"]
            diff = diff_map.get(iid, "medium")
            et = ERROR_TAXONOMY.get(task, {}).get(diff, "DKF")
            strict_counts[et] += 1

            if task == "contradiction_detection":
                is_residual = True  # CON not judge-reviewed; persists unconditionally
            else:
                jr = judged_rows.get(iid)
                is_residual = jr is not None and int(jr["judge_verdict"]) == 0
            if is_residual:
                residual_counts[et] += 1

        out[label] = {
            "strict": strict_counts, "strict_total": sum(strict_counts.values()),
            "residual": residual_counts, "residual_total": sum(residual_counts.values()),
        }
    return out


def fmt_pct_row(counts, total):
    order = ["DKF", "NRF", "TRF", "CGF"]
    cells = []
    for k in order:
        n = counts[k]
        pct = round(n / total * 100) if total else 0
        cells.append(f"{n} ({pct}\\%)")
    return cells


if __name__ == "__main__":
    con, yes_n, no_n, maj_base = con_balanced_accuracy()
    print(f"CON class balance: Yes={yes_n}, No={no_n}, majority baseline={maj_base:.1f}%\n")
    print(f"{'Model':<32}{'CON%':>7}{'vs.Base':>9}{'BalAcc':>8}")
    for label, d in con.items():
        vs_base = d["con_acc"] - maj_base
        print(f"{label:<32}{d['con_acc']:>7.1f}{vs_base:>+9.1f}{d['balanced_acc']:>8.1f}")

    Path("evaluation/con_balance_recomputed.json").write_text(
        json.dumps({"yes_n": yes_n, "no_n": no_n, "majority_baseline_pct": round(maj_base, 1),
                    "per_model": con}, indent=2), encoding="utf-8")

    BS = chr(92)
    lines = [BS + "begin{tabular}{lrrr}", BS + "toprule",
             BS + "textbf{Model} & " + BS + "textbf{CON \\%} & " + BS + "textbf{vs. Base} & " + BS + "textbf{Bal. Acc.} " + BS + BS,
             BS + "midrule"]
    for label, d in con.items():
        vs_base = d["con_acc"] - maj_base
        sign = "+" if vs_base >= 0 else "$-$"
        lines.append(f"{label} & {d['con_acc']:.1f} & {sign}{abs(vs_base):.1f} & {d['balanced_acc']:.1f} " + BS + BS)
    lines.append(BS + "midrule")
    lines.append(f"Majority baseline & {maj_base:.1f} & --- & 50.0 " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    Path("paper/tables/table_con_balance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    tax = error_taxonomy()
    for label, d in tax.items():
        print(f"\n{label}")
        print(f"  strict:   {d['strict']} total={d['strict_total']}")
        print(f"  residual: {d['residual']} total={d['residual_total']}")
        if d["strict_total"]:
            reclass = (d["strict_total"] - d["residual_total"]) / d["strict_total"] * 100
            print(f"  reclassified: {d['strict_total']-d['residual_total']}/{d['strict_total']} = {reclass:.1f}%")

    Path("evaluation/error_taxonomy_recomputed.json").write_text(
        json.dumps(tax, indent=2), encoding="utf-8")

    lines = [BS + "begin{tabular}{llrrrrr}", BS + "toprule",
             BS + "textbf{Model} & " + BS + "textbf{Errors} & " + BS + "textbf{DKF} & " +
             BS + "textbf{NRF} & " + BS + "textbf{TRF} & " + BS + "textbf{CGF} & " + BS + "textbf{Total} " + BS + BS,
             BS + "midrule"]
    for label, d in tax.items():
        s_cells = fmt_pct_row(d["strict"], d["strict_total"])
        r_cells = fmt_pct_row(d["residual"], d["residual_total"])
        lines.append(BS + f"multirow{{2}}{{*}}{{{label}}} & strict & " + " & ".join(s_cells) +
                     f" & {d['strict_total']} " + BS + BS)
        lines.append(f" & residual & " + " & ".join(r_cells) + f" & {d['residual_total']} " + BS + BS)
    lines += [BS + "bottomrule", BS + "end{tabular}"]
    Path("paper/tables/table_errortax.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWrote paper/tables/table_con_balance.tex and table_errortax.tex")
