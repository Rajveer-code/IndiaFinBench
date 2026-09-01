"""Analyze the full-coverage phi4-mini cross-model judge results.

Computes, per model: strict accuracy, phi4-judge-corrected accuracy, strict
false-negative rate (strict wrong, phi4 says right) and -- new, because this
judge covers strict-CORRECT items too -- strict false-positive rate (strict
right, phi4 disputes it). Also computes Gemini-vs-phi4 agreement on the
874 items the original Gemini judge reviewed, and evaluates the pre-committed
decision gate on DeepSeek-R1-Distill's reversal.

Outputs:
  evaluation/phi4_judge_analysis.json
  evaluation/phi4_regime_table.json  (feeds the rebuilt regime figure)
"""
import csv
import glob
import json
import os

MODEL_KEY_TO_LABEL = {
    "gemini": "Gemini 2.5 Flash", "gemini25_pro": "Gemini 2.5 Pro", "qwen3_32b": "Qwen3-32B",
    "groq70b": "LLaMA-3.3-70B", "llama4scout": "Llama 4 Scout 17B", "kimi_k2": "Kimi K2",
    "llama3": "LLaMA-3-8B", "gpt_oss_120b": "GPT-OSS 120B", "gpt_oss_20b": "GPT-OSS 20B",
    "mistral": "Mistral-7B", "deepseek_r1_70b": "DeepSeek-R1-Distill", "gemma4_e4b": "Gemma 3 4B",
}

per_model = {}
for path in sorted(glob.glob("evaluation/results_judged_phi4/*.csv")):
    key = os.path.basename(path).replace("_results.csv", "")
    label = MODEL_KEY_TO_LABEL[key]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 344, f"{label}: {len(rows)} rows, expected 344"

    strict_correct = sum(int(r["strict_correct"]) for r in rows)
    n = len(rows)

    # phi4-corrected: strict-correct OR phi4 says correct
    phi4_correct = sum(1 for r in rows
                        if int(r["strict_correct"]) == 1 or int(r["judge_verdict"]) == 1)

    # False negatives: strict wrong, phi4 says right
    fn = [r for r in rows if int(r["strict_correct"]) == 0 and int(r["judge_verdict"]) == 1]
    strict_wrong = [r for r in rows if int(r["strict_correct"]) == 0]
    fn_rate = len(fn) / len(strict_wrong) if strict_wrong else 0.0

    # False positives (NEW -- full coverage only): strict right, phi4 disputes
    fp = [r for r in rows if int(r["strict_correct"]) == 1 and int(r["judge_verdict"]) == 0]
    strict_right = [r for r in rows if int(r["strict_correct"]) == 1]
    fp_rate = len(fp) / len(strict_right) if strict_right else 0.0

    per_model[label] = {
        "n_judged_reg_num_tmp": n,
        "strict_correct_344": strict_correct,
        "phi4_corrected_344": phi4_correct,
        "fn_count": len(fn), "fn_rate": round(fn_rate, 4),
        "fp_count": len(fp), "fp_rate": round(fp_rate, 4),
    }

# Full 406-item accuracy: CON (62 items, exact match, unaffected by judge) + the 344 judged.
con_correct = {}
for path in sorted(glob.glob("evaluation/results/*_results.csv")):
    key = os.path.basename(path).replace("_results.csv", "")
    if key not in MODEL_KEY_TO_LABEL:
        continue
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    con_rows = [r for r in rows if r["task_type"] == "contradiction_detection"]
    con_correct[MODEL_KEY_TO_LABEL[key]] = sum(int(r["correct"]) for r in con_rows)

results = {}
for label, d in per_model.items():
    con = con_correct[label]
    strict_full = (d["strict_correct_344"] + con) / 406 * 100
    phi4_full = (d["phi4_corrected_344"] + con) / 406 * 100
    results[label] = {**d, "con_correct_62": con,
                       "strict_accuracy_406": round(strict_full, 2),
                       "phi4_corrected_accuracy_406": round(phi4_full, 2)}

# Rank + decision gate. Competition ranking (ties share the better rank) -- plain
# positional/enumerate ranking has silently broken a genuine tie here twice before
# in this project (once on the Gemini-based regime figure, once on an earlier pass
# of this exact script, patched only in the output JSON and never fixed at the
# source -- which is why the bug was still here to trip on a third time).
def competition_rank(vals: dict) -> dict:
    order = sorted(set(vals.values()), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(order)}
    return {k: rank_of[v] for k, v in vals.items()}


strict_vals = {label: d["strict_accuracy_406"] for label, d in results.items()}
phi4_vals = {label: d["phi4_corrected_accuracy_406"] for label, d in results.items()}
strict_rank = competition_rank(strict_vals)
phi4_rank = competition_rank(phi4_vals)
ranked_strict = sorted(results.items(), key=lambda x: -x[1]["strict_accuracy_406"])

deepseek_strict_rank = strict_rank["DeepSeek-R1-Distill"]
deepseek_phi4_rank = phi4_rank["DeepSeek-R1-Distill"]
tied_at_phi4_rank_1 = sorted(label for label, r in phi4_rank.items() if r == 1)
models_moved_2plus = sum(1 for label in results if abs(strict_rank[label] - phi4_rank[label]) >= 2)

print("=== Strict vs phi4-corrected accuracy (406-item basis) ===")
print(f"{'Model':22s} {'Strict':>8s} {'SRank':>6s} {'phi4-corr':>10s} {'PRank':>6s} {'FN%':>6s} {'FP%':>6s}")
for label, _ in ranked_strict:
    d = results[label]
    print(f"{label:22s} {d['strict_accuracy_406']:8.1f} {strict_rank[label]:6d} "
          f"{d['phi4_corrected_accuracy_406']:10.1f} {phi4_rank[label]:6d} "
          f"{100*d['fn_rate']:6.1f} {100*d['fp_rate']:6.1f}")

print(f"\nDeepSeek-R1-Distill: strict rank {deepseek_strict_rank} -> phi4-corrected rank {deepseek_phi4_rank}")
print(f"Tied at phi4 rank 1: {tied_at_phi4_rank_1}")
print(f"Models moved >=2 ranks: {models_moved_2plus}")
print(f"DECISION GATE: {'REVERSAL CONFIRMED' if deepseek_phi4_rank <= 2 and deepseek_strict_rank >= 10 else 'NOT CONFIRMED -- reassess spine'}")

json.dump({"per_model": results, "strict_rank": strict_rank, "phi4_rank": phi4_rank,
           "deepseek_strict_rank": deepseek_strict_rank, "deepseek_phi4_rank": deepseek_phi4_rank,
           "tied_at_phi4_rank_1": tied_at_phi4_rank_1, "models_moved_2plus": models_moved_2plus},
          open("evaluation/phi4_regime_table.json", "w"), indent=2)
print("\nSaved evaluation/phi4_regime_table.json")

# --- Gemini-vs-phi4 agreement on the 874 originally-Gemini-judged items ---
gemini_judged_dir = "evaluation/results_judged"
agree = disagree = 0
disagreements = []
gemini_key_map = {
    "gemini": "gemini", "gemini25_pro": "gemini25_pro", "qwen3_32b": "qwen3_32b",
    "groq70b": "groq70b", "llama4scout": "llama4scout", "kimi_k2": "kimi_k2",
    "llama3": "llama3", "gpt_oss_120b": "gpt_oss_120b", "gpt_oss_20b": "gpt_oss_20b",
    "mistral": "mistral", "deepseek_r1_70b": "deepseek_r1_70b", "gemma4_e4b": "gemma4_e4b",
}
for key, label in MODEL_KEY_TO_LABEL.items():
    gemini_path = f"{gemini_judged_dir}/{key}_results.csv"
    phi4_path = f"evaluation/results_judged_phi4/{key}_results.csv"
    if not os.path.exists(gemini_path):
        continue
    gemini_rows = {r["id"]: r for r in csv.DictReader(open(gemini_path, encoding="utf-8"))}
    phi4_rows = {r["id"]: r for r in csv.DictReader(open(phi4_path, encoding="utf-8"))}
    for iid, grow in gemini_rows.items():
        if grow.get("judge_score") in (None, "") or iid not in phi4_rows:
            continue
        g_verdict = int(grow["judge_score"]) if str(grow["judge_score"]).strip() != "" else None
        if g_verdict is None:
            continue
        p_verdict = int(phi4_rows[iid]["judge_verdict"])
        if g_verdict == p_verdict:
            agree += 1
        else:
            disagree += 1
            disagreements.append(iid)

total = agree + disagree
print(f"\n=== Gemini vs phi4-mini agreement (on Gemini's originally-judged item pool) ===")
print(f"agree={agree} disagree={disagree} total={total} raw_agreement={agree/total*100:.1f}%" if total else "no overlap found")
json.dump({"agree": agree, "disagree": disagree, "total": total,
           "raw_agreement_pct": round(agree / total * 100, 1) if total else None,
           "disagreement_ids_sample": disagreements[:50]},
          open("evaluation/gemini_vs_phi4_agreement.json", "w"), indent=2)
print("Saved evaluation/gemini_vs_phi4_agreement.json")
