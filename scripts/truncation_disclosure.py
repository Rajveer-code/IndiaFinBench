"""Plan v3 Phase 5.1 / F6: quantify logging truncation for the manuscript's disclosure
subsection. Computes, per model, the character cap the logging pipeline applied and how
many REG/NUM/TMP judged rows sit exactly at it, plus the truncation-free sensitivity
analysis (restricting to items where every model's prediction is strictly below its cap).

Outputs:
  evaluation/truncation_disclosure.json
"""
import glob
import json
import os

JUDGED_DIR = "evaluation/results_judged_phi4"

data = {}
caps = {}
for path in sorted(glob.glob(os.path.join(JUDGED_DIR, "*.csv"))):
    import csv
    model = os.path.basename(path).replace("_results.csv", "")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    data[model] = rows
    caps[model] = max(len(r["prediction"] or "") for r in rows)

n_total = sum(len(rows) for rows in data.values())
per_model = {}
n_at_cap = 0
for model, rows in data.items():
    at_cap = sum(1 for r in rows if len(r["prediction"] or "") >= caps[model])
    per_model[model] = {"n": len(rows), "cap": caps[model], "n_at_cap": at_cap,
                         "pct_at_cap": round(100 * at_cap / len(rows), 1)}
    n_at_cap += at_cap

# Truncation-free subset: items where every one of the 12 models' predictions is strictly
# below that model's own cap.
ids_sorted = sorted({r["id"] for r in next(iter(data.values()))})
model_by_id = {m: {r["id"]: r for r in rows} for m, rows in data.items()}
clean_ids = []
for iid in ids_sorted:
    ok = all(len(model_by_id[m][iid]["prediction"] or "") < caps[m] for m in data)
    if ok:
        clean_ids.append(iid)

def tf(v):
    return str(v).strip() == "1"

strict_clean = {}
augmented_clean = {}
for m, rows in data.items():
    by_id = model_by_id[m]
    s = sum(1 for i in clean_ids if tf(by_id[i]["strict_correct"]))
    a = sum(1 for i in clean_ids if tf(by_id[i]["strict_correct"]) or tf(by_id[i]["judge_verdict"]))
    strict_clean[m] = s
    augmented_clean[m] = a

n_clean = len(clean_ids)
strict_rank = {m: r + 1 for r, m in enumerate(sorted(strict_clean, key=lambda x: -strict_clean[x]))}
aug_rank = {m: r + 1 for r, m in enumerate(sorted(augmented_clean, key=lambda x: -augmented_clean[x]))}

out = {
    "n_judged_rows_total": n_total,
    "n_rows_at_cap": n_at_cap,
    "pct_rows_at_cap": round(100 * n_at_cap / n_total, 1),
    "per_model_caps": per_model,
    "judge_own_input_caps": {"question": 600, "reference": 300, "prediction": 500,
                              "judge_reason_output": 200,
                              "source": "scripts/judge_phi4_crossmodel.py:56,70"},
    "no_generation_truncation_record": True,
    "truncation_free_sensitivity": {
        "n_clean_items": n_clean, "n_total_items": len(ids_sorted),
        "deepseek_strict_pct": round(100 * strict_clean["deepseek_r1_70b"] / n_clean, 1),
        "deepseek_strict_rank": strict_rank["deepseek_r1_70b"],
        "deepseek_augmented_pct": round(100 * augmented_clean["deepseek_r1_70b"] / n_clean, 1),
        "deepseek_augmented_rank": aug_rank["deepseek_r1_70b"],
    },
}

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/truncation_disclosure.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"Pooled: {n_at_cap}/{n_total} = {out['pct_rows_at_cap']}% of judged rows at cap")
print("\nPer-model:")
for m, d in sorted(per_model.items(), key=lambda x: -x[1]["pct_at_cap"]):
    print(f"  {m:24s} cap={d['cap']:4d}  {d['n_at_cap']:3d}/{d['n']} ({d['pct_at_cap']}%)")
print(f"\nTruncation-free subset: {n_clean}/{len(ids_sorted)} REG/NUM/TMP items")
print(f"DeepSeek-R1-Distill on clean subset: strict={out['truncation_free_sensitivity']['deepseek_strict_pct']}% "
      f"(rank {out['truncation_free_sensitivity']['deepseek_strict_rank']}), "
      f"augmented={out['truncation_free_sensitivity']['deepseek_augmented_pct']}% "
      f"(rank {out['truncation_free_sensitivity']['deepseek_augmented_rank']})")
print("Saved evaluation/truncation_disclosure.json")
