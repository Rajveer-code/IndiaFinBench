"""
judge_corrected_analysis.py
---------------------------
Purpose:  Reproduce every judge-correction number in the paper from the
          per-item audit log:
            1. Judge-corrected accuracies per model x task (Appendix D, Table D1)
            2. Exclusion accounting (935 flagged = 874 judged + 58 empty + 3 FAIL)
            3. Paired bootstrap p-values for all 66 model pairs under both
               scoring regimes (strict and judge-corrected)
            4. Strict vs residual error taxonomy (Section 6, Table 4)
Inputs:   evaluation/results/{model}_results.csv
          evaluation/results_judged/judge_audit_log.csv
Outputs:  evaluation/error_analysis/p_matrix_full.csv
          evaluation/error_analysis/judge_corrected_accuracies.csv
          evaluation/error_analysis/error_taxonomy_strict_residual.csv
          Formatted console report
Usage:
    python scripts/judge_corrected_analysis.py
"""
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("evaluation/results")
JUDGE_LOG = Path("evaluation/results_judged/judge_audit_log.csv")
OUT_DIR = Path("evaluation/error_analysis")

MODELS = {  # file stem -> paper display name (Table 1 order)
    "gemini":          "Gemini 2.5 Flash",
    "qwen3_32b":       "Qwen3-32B",
    "groq70b":         "LLaMA-3.3-70B",
    "llama4scout":     "Llama 4 Scout 17B",
    "kimi_k2":         "Kimi K2",
    "llama3":          "LLaMA-3-8B",
    "gpt_oss_120b":    "GPT-OSS 120B",
    "gpt_oss_20b":     "GPT-OSS 20B",
    "gemini25_pro":    "Gemini 2.5 Pro",
    "mistral":         "Mistral-7B",
    "deepseek_r1_70b": "DeepSeek R1 70B",
    "gemma4_e4b":      "Gemma 4 E4B",
}
TASKS = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}
# Error-type mapping (Section 6): task type + difficulty -> error type.
ERROR_TAXONOMY = {
    "regulatory_interpretation": {"easy": "DKF", "medium": "DKF", "hard": "CGF"},
    "numerical_reasoning":       {"easy": "NRF", "medium": "NRF", "hard": "NRF"},
    "contradiction_detection":   {"easy": "TRF", "medium": "TRF", "hard": "TRF"},
    "temporal_reasoning":        {"easy": "TRF", "medium": "TRF", "hard": "DKF"},
}
N_RESAMPLES = 10_000
SEED = 42


def load_model(stem: str) -> dict[str, dict]:
    with (RESULTS_DIR / f"{stem}_results.csv").open(encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired bootstrap (shift method), matching
    scripts/bootstrap_significance.py."""
    n = len(a)
    obs = a.mean() - b.mean()
    centred = (a - b) - (a - b).mean()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    boot = centred[idx].mean(axis=1)
    return float(np.mean(np.abs(boot) >= np.abs(obs)))


def main() -> None:
    data = {MODELS[s]: load_model(s) for s in MODELS}
    flips: dict[str, dict[str, bool]] = defaultdict(dict)
    with JUDGE_LOG.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            flips[r["model"]][r["id"]] = r["flipped"].strip().lower() == "true"

    ids_sorted = sorted(next(iter(data.values())).keys())

    # 1. corrected accuracies -----------------------------------------------
    acc_rows = []
    for m, rows in data.items():
        per = {"strict": defaultdict(lambda: [0, 0]), "corr": defaultdict(lambda: [0, 0])}
        for iid, r in rows.items():
            t = TASKS[r["task_type"]]
            c = int(r["correct"] or 0)
            cc = 1 if (c == 1 or flips[m].get(iid, False)) else 0
            for key, val in (("strict", c), ("corr", cc)):
                for bucket in (t, "ALL"):
                    per[key][bucket][0] += val
                    per[key][bucket][1] += 1
        row = {"model": m}
        for t in ("REG", "NUM", "CON", "TMP", "ALL"):
            row[f"{t}_strict"] = round(100 * per["strict"][t][0] / per["strict"][t][1], 1)
            row[f"{t}_corrected"] = round(100 * per["corr"][t][0] / per["corr"][t][1], 1)
        acc_rows.append(row)

    # 2. exclusion accounting ------------------------------------------------
    excl = {"incorrect": 0, "judged": 0, "empty": 0, "fail": 0}
    for m, rows in data.items():
        for iid, r in rows.items():
            if TASKS[r["task_type"]] == "CON" or int(r["correct"] or 0) == 1:
                continue
            excl["incorrect"] += 1
            pred = (r.get("prediction") or "")
            if iid in flips[m]:
                excl["judged"] += 1
            elif not pred.strip():
                excl["empty"] += 1
            elif "FAIL" in pred:
                excl["fail"] += 1

    # 3. p-matrix under both regimes ----------------------------------------
    vec = {}
    for m, rows in data.items():
        vec[m] = {
            "strict": np.array([int(rows[i]["correct"] or 0) for i in ids_sorted]),
            "corr": np.array([
                1 if (int(rows[i]["correct"] or 0) == 1 or flips[m].get(i, False)) else 0
                for i in ids_sorted
            ]),
        }
    p_rows = []
    for a, b in combinations(MODELS.values(), 2):
        p_rows.append({
            "model_a": a, "model_b": b,
            "p_strict": paired_bootstrap(vec[a]["strict"], vec[b]["strict"]),
            "p_corrected": paired_bootstrap(vec[a]["corr"], vec[b]["corr"]),
        })

    # 4. strict vs residual taxonomy ----------------------------------------
    tax_rows = []
    for m in MODELS.values():
        strict = defaultdict(int)
        resid = defaultdict(int)
        for i in ids_sorted:
            r = data[m][i]
            if int(r["correct"] or 0) == 0:
                et = ERROR_TAXONOMY[r["task_type"]][r["difficulty"].lower()]
                strict[et] += 1
                if not flips[m].get(i, False):
                    resid[et] += 1
        for regime, d in (("strict", strict), ("residual", resid)):
            tax_rows.append({"model": m, "regime": regime,
                             **{et: d[et] for et in ("DKF", "NRF", "TRF", "CGF")},
                             "total": sum(d.values())})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("judge_corrected_accuracies.csv", acc_rows),
                       ("p_matrix_full.csv", p_rows),
                       ("error_taxonomy_strict_residual.csv", tax_rows)):
        path = OUT_DIR / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {path}")

    n_sig = sum(1 for r in p_rows if r["p_strict"] < 0.05)
    n_bonf = sum(1 for r in p_rows if r["p_strict"] < 0.05 / 66)
    n_pres = sum(1 for r in p_rows if r["p_strict"] < 0.05 and r["p_corrected"] < 0.05)
    print(f"\nFlagged incorrect (non-CON): {excl['incorrect']} = "
          f"{excl['judged']} judged + {excl['empty']} empty + {excl['fail']} API-FAIL")
    print(f"Pairs significant at 0.05 (strict): {n_sig}/66; "
          f"Bonferroni: {n_bonf}/66; preserved under correction: {n_pres}")


if __name__ == "__main__":
    main()
