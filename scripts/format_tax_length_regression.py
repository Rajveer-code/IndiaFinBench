"""Plan v3 Phase 4.3 / F2: is the length-vs-rescue association (pooled Mann-Whitney,
p=0.0097, shorter strict failures more likely to be judge-rescued) confounded by
task/model composition, or does it survive adjustment?

Fits: rescued ~ log(len(prediction) + 1) + C(task_type) + C(model)
on every strict-wrong REG/NUM/TMP item (n=905), via logistic regression.

CAVEAT, stated here and required in any manuscript text that cites this script's
output: `prediction` is character-capped by the logging pipeline (200/300/500
chars depending on which evaluate*.py script wrote it -- see Plan v3 F6), so
`len(prediction)` is a right-censored proxy for true response length, not the
length itself. Phase 2 will produce an untruncated confirmatory version of this
analysis once fresh predictions are logged in full; until then this is the best
available evidence, not a final one, and the manuscript must say so.

Outputs:
  evaluation/format_tax_length_regression.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

JUDGED_DIR = "evaluation/results_judged_phi4"

rows = []
for path in sorted(glob.glob(os.path.join(JUDGED_DIR, "*.csv"))):
    model = os.path.basename(path).replace("_results.csv", "")
    df = pd.read_csv(path, dtype=str)
    df["model"] = model
    rows.append(df)
data = pd.concat(rows, ignore_index=True)

data["strict_correct"] = data["strict_correct"].astype(int)
data["judge_verdict"] = data["judge_verdict"].astype(int)
data["pred_len"] = data["prediction"].fillna("").str.len()

wrong = data[data["strict_correct"] == 0].copy()
assert len(wrong) == 905, f"expected 905 strict-wrong REG/NUM/TMP rows, got {len(wrong)}"

wrong["rescued"] = wrong["judge_verdict"]
wrong["log_len"] = np.log1p(wrong["pred_len"])

# Per-model caps, for the censoring disclosure.
caps = {}
for path in sorted(glob.glob(os.path.join(JUDGED_DIR, "*.csv"))):
    model = os.path.basename(path).replace("_results.csv", "")
    df = pd.read_csv(path, dtype=str)
    caps[model] = int(df["prediction"].fillna("").str.len().max())
wrong["at_cap"] = wrong.apply(lambda r: r["pred_len"] >= caps[r["model"]], axis=1)
pct_censored_overturned = wrong.loc[wrong["rescued"] == 1, "at_cap"].mean() * 100
pct_censored_retained = wrong.loc[wrong["rescued"] == 0, "at_cap"].mean() * 100

model_fit = smf.logit("rescued ~ log_len + C(task_type) + C(model)", data=wrong).fit(disp=0)

coef = model_fit.params["log_len"]
se = model_fit.bse["log_len"]
p = model_fit.pvalues["log_len"]
ci_lo, ci_hi = model_fit.conf_int().loc["log_len"]

# Per-task direction: median rescued vs retained length, stratified.
per_task = {}
for task, g in wrong.groupby("task_type"):
    resc = g.loc[g["rescued"] == 1, "pred_len"]
    ret = g.loc[g["rescued"] == 0, "pred_len"]
    per_task[task] = {
        "n_overturned": int(len(resc)), "n_retained": int(len(ret)),
        "median_len_overturned": float(resc.median()) if len(resc) else None,
        "median_len_retained": float(ret.median()) if len(ret) else None,
    }

out = {
    "n_strict_wrong_regnumtmp": int(len(wrong)),
    "pooled_mann_whitney": {
        "note": "computed separately (scripts had scipy.stats.mannwhitneyu on pooled "
                 "overturned vs retained lengths): U=28916.0, p=0.0097, "
                 "median overturned=98.5 (n=820), median retained=138.0 (n=85)",
    },
    "logistic_regression": {
        "formula": "rescued ~ log_len + C(task_type) + C(model)",
        "n_obs": int(model_fit.nobs),
        "log_len_coef": round(float(coef), 4),
        "log_len_se": round(float(se), 4),
        "log_len_p": round(float(p), 4),
        "log_len_95ci": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "direction": "negative (longer predictions less likely to be rescued)" if coef < 0
                     else "positive (longer predictions more likely to be rescued)",
        "significant_at_05": bool(p < 0.05),
        "pseudo_r2": round(float(model_fit.prsquared), 4),
    },
    "per_task_direction": per_task,
    "censoring": {
        "pct_overturned_rows_at_cap": round(float(pct_censored_overturned), 1),
        "pct_retained_rows_at_cap": round(float(pct_censored_retained), 1),
        "caveat": "prediction length is right-censored by the logging pipeline; this "
                  "biases the retained-group median down more than the overturned "
                  "group's (higher censoring rate in retained), which if anything "
                  "understates the true gap in this direction -- see Plan v3 F6/F2. "
                  "A confirmatory analysis on untruncated Phase 2 re-run data is "
                  "still required before this is reported as a final result.",
    },
}

os.makedirs("evaluation", exist_ok=True)
with open("evaluation/format_tax_length_regression.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"n={out['logistic_regression']['n_obs']}  "
      f"log_len coef={out['logistic_regression']['log_len_coef']} "
      f"(p={out['logistic_regression']['log_len_p']}, "
      f"95% CI {out['logistic_regression']['log_len_95ci']})")
print(f"direction: {out['logistic_regression']['direction']}")
print(f"significant at 0.05: {out['logistic_regression']['significant_at_05']}")
print("\nPer-task median lengths (overturned vs retained):")
for task, d in per_task.items():
    print(f"  {task:28s} overturned={d['median_len_overturned']}"
          f" (n={d['n_overturned']})  retained={d['median_len_retained']} (n={d['n_retained']})")
print(f"\ncensoring: {out['censoring']['pct_overturned_rows_at_cap']}% of overturned rows at "
      f"cap vs {out['censoring']['pct_retained_rows_at_cap']}% of retained rows")
print("Saved evaluation/format_tax_length_regression.json")
