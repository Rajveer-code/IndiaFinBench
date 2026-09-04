"""Judge-tolerance sensitivity: compare judge-only and judge-augmented regimes across
three rounding tolerances (exact, 0.1%, 1% -- the existing full-coverage run) to test
whether the paper's headline findings depend on the rubric's "rounding within 1%" clause.

Reuses regime_table.py's exact methodology (same CON handling, same competition ranking)
against three different results_judged_phi4* directories instead of duplicating it.

Outputs: evaluation/judge_tolerance_comparison.json
"""
import csv
import glob
import json
import os

from scipy.stats import kendalltau, spearmanr

MODEL_KEY_TO_LABEL = {
    "gemini": "Gemini 2.5 Flash", "gemini25_pro": "Gemini 2.5 Pro", "qwen3_32b": "Qwen3-32B",
    "groq70b": "LLaMA-3.3-70B", "llama4scout": "Llama 4 Scout 17B", "kimi_k2": "Kimi K2",
    "llama3": "LLaMA-3-8B", "gpt_oss_120b": "GPT-OSS 120B", "gpt_oss_20b": "GPT-OSS 20B",
    "mistral": "Mistral-7B", "deepseek_r1_70b": "DeepSeek-R1-Distill", "gemma4_e4b": "Gemma 3 4B",
}

VARIANTS = {
    "exact": "evaluation/results_judged_phi4_tol_exact",
    "0.1pct": "evaluation/results_judged_phi4_tol_0_1pct",
    "1pct": "evaluation/results_judged_phi4",  # existing full-coverage run, reused as-is
}


def competition_rank(vals: dict) -> dict:
    order = sorted(set(vals.values()), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of[v] for k, v in vals.items()}


def load_con_correct():
    out = {}
    for path in sorted(glob.glob("evaluation/results/*_results.csv")):
        key = os.path.basename(path).replace("_results.csv", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r.get("task_type") == "contradiction_detection"]
        out[key] = sum(int(r["correct"]) for r in rows)
    return out


def compute_variant(judged_dir: str, con: dict) -> dict:
    per_model = {}
    for path in sorted(glob.glob(os.path.join(judged_dir, "*.csv"))):
        key = os.path.basename(path).replace("_results.csv", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        assert len(rows) == 344, f"{judged_dir}/{key}: {len(rows)} rows, expected 344"
        label = MODEL_KEY_TO_LABEL[key]
        strict_344 = sum(1 for r in rows if r["strict_correct"] == "1")
        judge_only_344 = sum(1 for r in rows if r["judge_verdict"] == "1")
        augmented_344 = sum(1 for r in rows
                             if r["strict_correct"] == "1" or r["judge_verdict"] == "1")
        c = con[key]
        n406 = 344 + 62
        per_model[label] = {
            "strict_pct": round((strict_344 + c) / n406 * 100, 2),
            "judge_only_pct": round((judge_only_344 + c) / n406 * 100, 2),
            "judge_augmented_pct": round((augmented_344 + c) / n406 * 100, 2),
        }

    strict_vals = {m: d["strict_pct"] for m, d in per_model.items()}
    judge_only_vals = {m: d["judge_only_pct"] for m, d in per_model.items()}
    augmented_vals = {m: d["judge_augmented_pct"] for m, d in per_model.items()}
    strict_rank = competition_rank(strict_vals)
    judge_only_rank = competition_rank(judge_only_vals)
    augmented_rank = competition_rank(augmented_vals)
    for m in per_model:
        per_model[m]["strict_rank"] = strict_rank[m]
        per_model[m]["judge_only_rank"] = judge_only_rank[m]
        per_model[m]["judge_augmented_rank"] = augmented_rank[m]

    models = list(per_model)
    s = [strict_vals[m] for m in models]
    j = [judge_only_vals[m] for m in models]
    rho, p_rho = spearmanr(s, j)
    tau, p_tau = kendalltau(s, j)

    return {
        "per_model": per_model,
        "judge_only_spread_pp": round(max(j) - min(j), 2),
        "judge_augmented_spread_pp": round(max(augmented_vals.values()) - min(augmented_vals.values()), 2),
        "strict_vs_judge_only_spearman_rho": round(float(rho), 4),
        "strict_vs_judge_only_spearman_p": round(float(p_rho), 4),
        "strict_vs_judge_only_kendall_tau": round(float(tau), 4),
        "strict_vs_judge_only_kendall_p": round(float(p_tau), 4),
    }


def main():
    con = load_con_correct()
    results = {name: compute_variant(path, con) for name, path in VARIANTS.items()}

    print(f"{'Tolerance':10s} {'judge-only spread':>18s} {'judge-aug spread':>17s} "
          f"{'rho(strict,jo)':>15s} {'p':>7s}")
    for name, d in results.items():
        print(f"{name:10s} {d['judge_only_spread_pp']:18.2f} {d['judge_augmented_spread_pp']:17.2f} "
              f"{d['strict_vs_judge_only_spearman_rho']:+15.3f} {d['strict_vs_judge_only_spearman_p']:7.3f}")

    print("\n=== DeepSeek-R1-Distill across tolerances ===")
    print(f"{'Tolerance':10s} {'judge-only%':>12s} {'JO rank':>8s} {'judge-aug%':>11s} {'JA rank':>8s}")
    for name, d in results.items():
        ds = d["per_model"]["DeepSeek-R1-Distill"]
        print(f"{name:10s} {ds['judge_only_pct']:12.2f} {ds['judge_only_rank']:8d} "
              f"{ds['judge_augmented_pct']:11.2f} {ds['judge_augmented_rank']:8d}")

    print("\n=== Every model's judge-only rank across tolerances (rank movement) ===")
    print(f"{'Model':22s} {'exact':>7s} {'0.1pct':>7s} {'1pct':>7s} {'max_move':>9s}")
    for m in MODEL_KEY_TO_LABEL.values():
        ranks = [results[v]["per_model"][m]["judge_only_rank"] for v in ("exact", "0.1pct", "1pct")]
        move = max(ranks) - min(ranks)
        print(f"{m:22s} {ranks[0]:7d} {ranks[1]:7d} {ranks[2]:7d} {move:9d}")

    with open("evaluation/judge_tolerance_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved evaluation/judge_tolerance_comparison.json")

    # LaTeX table: judge-only %/rank at each tolerance, sorted by the 1pct (paper's) rank.
    order = sorted(MODEL_KEY_TO_LABEL.values(),
                    key=lambda m: results["1pct"]["per_model"][m]["judge_only_rank"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Exact}} & \multicolumn{2}{c}{\textbf{0.1\%}} & "
        r"\multicolumn{2}{c}{\textbf{1\% (paper)}} \\",
        r"\textbf{Model} & \textbf{Judge-only} & \textbf{Rank} & \textbf{Judge-only} & "
        r"\textbf{Rank} & \textbf{Judge-only} & \textbf{Rank} \\",
        r"\midrule",
    ]
    for m in order:
        bold = m == "DeepSeek-R1-Distill"
        name = rf"\textbf{{{m}}}" if bold else m
        cells = []
        for v in ("exact", "0.1pct", "1pct"):
            d = results[v]["per_model"][m]
            cells += [f"{d['judge_only_pct']:.2f}", str(d["judge_only_rank"])]
        row = " & ".join(f"\\textbf{{{c}}}" if bold else c for c in cells)
        lines.append(f"{name} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/table_judgetolerance.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote paper/tables/table_judgetolerance.tex")


if __name__ == "__main__":
    main()
