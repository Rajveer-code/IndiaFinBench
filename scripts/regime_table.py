"""Single source of truth for the paper's three scoring regimes (Plan v3, Phase 0.1).

Computes, per model, from the same underlying predictions:
  - strict       -- the four-stage string-matching pipeline (primary regime)
  - judge_only   -- phi4-mini's verdict is final, in both directions (primary regime)
  - judge_augmented -- strict OR judge (operational composite / sensitivity regime;
                       this is what earlier drafts called "judge-audited")

CON (62 items/model) is never judge-reviewed (Section 4.3 rationale) and is scored
strictly in all three regimes, exactly as in the existing pipeline
(scripts/analyze_phi4_judge.py). REG/NUM/TMP (344 items/model) come from
evaluation/results_judged_phi4/*.csv, which carries both strict_correct and
judge_verdict per item.

Reuses MODEL_KEY_TO_LABEL and competition_rank from analyze_phi4_judge.py rather
than redefining them -- see CLAUDE.md Guardrail 5 (fix at the source, one
definition per concept).

Outputs:
  evaluation/regime_three_way.json
  paper/tables/table_regime.tex   (must be manually copied to
    paper/tmlr/tmlr_submission/tables/ -- see plan Guardrail 6)
"""
import csv
import glob
import json
import os

# MODEL_KEY_TO_LABEL and competition_rank are intentionally duplicated from
# analyze_phi4_judge.py, not imported: that script has no `if __name__ ==
# "__main__":` guard and executes its full body (file reads, prints, writes
# to evaluation/phi4_regime_table.json and evaluation/gemini_vs_phi4_agreement.json)
# at import time. Importing it here for two small definitions would silently
# trigger all of that as a side effect. Same duplication pattern already used
# by scripts/build_adjudication_control_sample.py for the same reason -- keep
# both copies in sync if the canonical model roster ever changes.
MODEL_KEY_TO_LABEL = {
    "gemini": "Gemini 2.5 Flash", "gemini25_pro": "Gemini 2.5 Pro", "qwen3_32b": "Qwen3-32B",
    "groq70b": "LLaMA-3.3-70B", "llama4scout": "Llama 4 Scout 17B", "kimi_k2": "Kimi K2",
    "llama3": "LLaMA-3-8B", "gpt_oss_120b": "GPT-OSS 120B", "gpt_oss_20b": "GPT-OSS 20B",
    "mistral": "Mistral-7B", "deepseek_r1_70b": "DeepSeek-R1-Distill", "gemma4_e4b": "Gemma 3 4B",
}


def competition_rank(vals: dict) -> dict:
    order = sorted(set(vals.values()), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of[v] for k, v in vals.items()}


try:
    from scipy.stats import spearmanr, kendalltau
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

JUDGED_DIR = "evaluation/results_judged_phi4"
RESULTS_DIR = "evaluation/results"


def load_judged():
    """key -> list of rows from results_judged_phi4/<key>_results.csv (344 rows, REG/NUM/TMP)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(JUDGED_DIR, "*.csv"))):
        key = os.path.basename(path).replace("_results.csv", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue
        out[key] = list(csv.DictReader(open(path, encoding="utf-8")))
    return out


def load_con_correct():
    """key -> int, CON-task correct count out of 62, from evaluation/results/<key>_results.csv."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.csv"))):
        key = os.path.basename(path).replace("_results.csv", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r.get("task_type") == "contradiction_detection"]
        out[key] = sum(int(r["correct"]) for r in rows)
    return out


def main():
    judged = load_judged()
    con = load_con_correct()

    missing_judged = set(MODEL_KEY_TO_LABEL) - set(judged)
    missing_con = set(MODEL_KEY_TO_LABEL) - set(con)
    if missing_judged or missing_con:
        raise SystemExit(f"missing data -- judged:{missing_judged} con:{missing_con}")

    per_model = {}
    for key, label in MODEL_KEY_TO_LABEL.items():
        rows = judged[key]
        n344 = len(rows)
        strict_344 = sum(1 for r in rows if r["strict_correct"] == "1")
        judge_only_344 = sum(1 for r in rows if r["judge_verdict"] == "1")
        augmented_344 = sum(1 for r in rows
                             if r["strict_correct"] == "1" or r["judge_verdict"] == "1")
        c = con[key]
        n406 = n344 + 62
        per_model[label] = {
            "key": key,
            "n_344": n344,
            "strict_344": strict_344,
            "judge_only_344": judge_only_344,
            "judge_augmented_344": augmented_344,
            "con_correct_62": c,
            "strict_406": strict_344 + c,
            "judge_only_406": judge_only_344 + c,
            "judge_augmented_406": augmented_344 + c,
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
    a = [augmented_vals[m] for m in models]

    correlations = {}
    if HAVE_SCIPY:
        for name, x, y in [
            ("strict_vs_judge_only", s, j),
            ("strict_vs_judge_augmented", s, a),
            ("judge_only_vs_judge_augmented", j, a),
        ]:
            rho, p_s = spearmanr(x, y)
            tau, p_t = kendalltau(x, y)
            correlations[name] = {
                "spearman_rho": round(float(rho), 4), "spearman_p": round(float(p_s), 4),
                "kendall_tau": round(float(tau), 4), "kendall_p": round(float(p_t), 4),
            }

    out = {
        "n_models": len(per_model),
        "per_model": per_model,
        "spread": {
            "strict_pp": round(max(s) - min(s), 2),
            "judge_only_pp": round(max(j) - min(j), 2),
            "judge_augmented_pp": round(max(a) - min(a), 2),
        },
        "correlations": correlations,
        "note": "judge_augmented is an asymmetric operational composite (strict OR judge), "
                "not a primary regime -- see plan v3 Section 3.",
    }

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/regime_three_way.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # LaTeX table, sorted by strict rank (matches existing table_regime.tex convention)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Strict & Rank & Judge-only & Rank & Judge-augmented & Rank \\",
        r"\midrule",
    ]
    for m in sorted(per_model, key=lambda x: per_model[x]["strict_rank"]):
        d = per_model[m]
        bold = m == "DeepSeek-R1-Distill"
        name = rf"\textbf{{{m}}}" if bold else m
        vals = [d["strict_pct"], d["strict_rank"], d["judge_only_pct"], d["judge_only_rank"],
                d["judge_augmented_pct"], d["judge_augmented_rank"]]
        cells = " & ".join(f"\\textbf{{{v}}}" if bold else str(v) for v in vals)
        lines.append(f"{name} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/table_regime.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"n_models={out['n_models']}  spread strict={out['spread']['strict_pp']}  "
          f"judge_only={out['spread']['judge_only_pp']}  "
          f"judge_augmented={out['spread']['judge_augmented_pp']}")
    if HAVE_SCIPY:
        for name, c in correlations.items():
            print(f"  {name}: rho={c['spearman_rho']:+.3f} (p={c['spearman_p']:.3f})  "
                  f"tau={c['kendall_tau']:+.3f} (p={c['kendall_p']:.3f})")
    print("Wrote evaluation/regime_three_way.json and paper/tables/table_regime.tex")
    print("REMINDER: copy paper/tables/table_regime.tex -> "
          "paper/tmlr/tmlr_submission/tables/table_regime.tex before compiling.")


if __name__ == "__main__":
    main()
