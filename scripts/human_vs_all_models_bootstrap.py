"""All 12 human-vs-model paired bootstrap tests on the 60-item human-eval
subset, with Bonferroni correction across the 12-test family.

Reuses score_human_eval.py's scorer and generate_figures.py's
bootstrap_pvalue exactly (same 10,000-resample paired bootstrap already used
throughout the paper) rather than inventing new methodology. Previously only
2 of the 12 comparisons (best and worst model) were computed and reported,
with no multiplicity correction across the implicit 12-test family raised by
reporting the extremes.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score_human_eval import ref_map, rows as human_rows, score_item  # noqa: E402
from novel_methods_utils import MODEL_FILES, RESULTS_DIR  # noqa: E402


def bootstrap_pvalue(correct_a, correct_b, n_resamples=10_000):
    diff = np.array(correct_a) - np.array(correct_b)
    obs = np.mean(diff)
    centred = diff - np.mean(diff)
    n = len(centred)
    count = 0
    for _ in range(n_resamples):
        sample = centred[np.random.randint(0, n, n)]
        if abs(np.mean(sample)) >= abs(obs):
            count += 1
    return count / n_resamples


human_ids = []
human_correct = []
for row in human_rows:
    iid = row["id"]
    ref_item = ref_map.get(iid)
    ans = row.get("your_answer", "").strip()
    if not ref_item or not ans:
        continue
    c, _ = score_item(ref_item["answer"], ans, row["task_type"])
    human_ids.append(iid)
    human_correct.append(c)

assert len(human_ids) == 60, f"expected 60 human items, got {len(human_ids)}"
human_acc = sum(human_correct) / len(human_correct) * 100

results = {}
for label, fname in MODEL_FILES.items():
    model_rows = {r["id"]: int(r["correct"]) for r in csv.DictReader(open(RESULTS_DIR / fname, encoding="utf-8"))}
    model_correct = [model_rows[iid] for iid in human_ids]
    model_acc = sum(model_correct) / len(model_correct) * 100
    p = bootstrap_pvalue(human_correct, model_correct)
    results[label] = {"model_acc_60item": round(model_acc, 1), "human_acc_60item": round(human_acc, 1),
                       "p_value": round(p, 4)}

n_tests = len(results)
alpha = 0.05
bonf_alpha = alpha / n_tests
print(f"Human accuracy on 60-item subset: {human_acc:.1f}%")
print(f"n_tests = {n_tests}, Bonferroni alpha = {alpha}/{n_tests} = {bonf_alpha:.5f}\n")
print(f"{'Model':22s} {'Acc':>7s} {'p':>8s} {'sig(.05)':>9s} {'sig(Bonf)':>10s}")
for label, d in sorted(results.items(), key=lambda x: -x[1]["model_acc_60item"]):
    p = d["p_value"]
    sig05 = "yes" if p < 0.05 else "no"
    sigb = "yes" if p < bonf_alpha else "no"
    results[label]["sig_p05"] = sig05 == "yes"
    results[label]["sig_bonferroni"] = sigb == "yes"
    print(f"{label:22s} {d['model_acc_60item']:7.1f} {p:8.4f} {sig05:>9s} {sigb:>10s}")

n_sig05 = sum(1 for d in results.values() if d["sig_p05"])
n_sigbonf = sum(1 for d in results.values() if d["sig_bonferroni"])
print(f"\nSignificant at p<0.05 (uncorrected): {n_sig05}/{n_tests}")
print(f"Significant after Bonferroni (n={n_tests}): {n_sigbonf}/{n_tests}")

json.dump({"human_acc_60item": round(human_acc, 1), "n_tests": n_tests, "bonferroni_alpha": round(bonf_alpha, 5),
           "per_model": results, "n_sig_uncorrected": n_sig05, "n_sig_bonferroni": n_sigbonf},
          open("evaluation/human_vs_all_models_bootstrap.json", "w"), indent=2)
print("\nSaved evaluation/human_vs_all_models_bootstrap.json")
