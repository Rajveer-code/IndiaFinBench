"""Extend the human-adjudication sheet with a stratified control sample.

The existing 174-row sheet (annotation/judge_audit/adjudication_sheet.csv) is
every REG/NUM/TMP item where Gemini 2.5 Flash and phi4-mini disagreed -- all
of it drawn from strict-INCORRECT items, since Gemini only ever judged strict
failures. It has no strict-correct items and no CON items at all.

This script adds a small control sample of cases where the scoring methods
AGREE, stratified across {strict-correct, strict-incorrect} x all four task
types, so the human adjudicator can check whether agreement between methods
also means agreement with a human reader -- not to estimate a population-level
judge error rate (174 + ~64 items is nowhere near enough for that), but as
direct evidence of where automated scoring and human judgment do or don't
line up.

Strata (N_PER_STRATUM sampled from each, fixed seed for reproducibility):
  REG/NUM/TMP, strict-incorrect, Gemini & phi4-mini agree  (both judges concur)
  REG/NUM/TMP, strict-correct,   phi4-mini agrees (says correct too)
  CON,         strict-incorrect  (no judge exists for CON; direct spot-check)
  CON,         strict-correct    (no judge exists for CON; direct spot-check)

Output: annotation/judge_audit/adjudication_sheet.csv (overwritten, now with
a `sample_group` and `stratum` column), old 174 rows preserved verbatim.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_JUDGED = ROOT / "evaluation" / "results_judged"
RESULTS_JUDGED_PHI4 = ROOT / "evaluation" / "results_judged_phi4"
SHEET_PATH = ROOT / "annotation" / "judge_audit" / "adjudication_sheet.csv"

N_PER_STRATUM = 8
SEED = 42

MODEL_KEY_TO_LABEL = {
    "gemini": "Gemini 2.5 Flash", "gemini25_pro": "Gemini 2.5 Pro", "qwen3_32b": "Qwen3-32B",
    "groq70b": "LLaMA-3.3-70B", "llama4scout": "Llama 4 Scout 17B", "kimi_k2": "Kimi K2",
    "llama3": "LLaMA-3-8B", "gpt_oss_120b": "GPT-OSS 120B", "gpt_oss_20b": "GPT-OSS 20B",
    "mistral": "Mistral-7B", "deepseek_r1_70b": "DeepSeek-R1-Distill", "gemma4_e4b": "Gemma 4 E4B",
}
TASK_LABEL = {
    "regulatory_interpretation": "REG", "numerical_reasoning": "NUM",
    "temporal_reasoning": "TMP", "contradiction_detection": "CON",
}

FIELDNAMES = ["model", "item_id", "task_type", "strict_correct", "gemini_verdict",
              "phi4_verdict", "question", "reference_answer", "model_prediction",
              "sample_group", "stratum",
              "human_verdict_CORRECT_or_INCORRECT", "human_notes"]


def load_existing():
    with open(SHEET_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["sample_group"] = "disagreement"
        r["stratum"] = f"{TASK_LABEL.get(r['task_type'], r['task_type'])}_disagreement"
        for k in FIELDNAMES:
            r.setdefault(k, "")
    return rows


def build_pools():
    """Returns dict: stratum_name -> list of candidate row dicts."""
    pools = {}

    def add(stratum, row):
        pools.setdefault(stratum, []).append(row)

    # REG/NUM/TMP: need strict + gemini + phi4 verdicts joined per (model, id).
    for path in sorted(RESULTS_JUDGED.glob("*.csv")):
        key = path.stem.replace("_results", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue  # skip the 2 extra non-canonical files in this dir
        label = MODEL_KEY_TO_LABEL[key]
        phi4_path = RESULTS_JUDGED_PHI4 / path.name
        phi4_by_id = {r["id"]: r for r in csv.DictReader(open(phi4_path, newline="", encoding="utf-8"))} \
            if phi4_path.exists() else {}

        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            task = r["task_type"]
            if task == "contradiction_detection":
                continue  # handled separately below, no judge involved
            tshort = TASK_LABEL[task]
            strict_ok = r["correct"] == "1"
            base = {
                "model": label, "item_id": r["id"], "task_type": task,
                "strict_correct": r["correct"],
                "question": r["question"], "reference_answer": r["ref_answer"],
                "model_prediction": r["prediction"],
            }
            if not strict_ok:
                # Gemini only judged strict-incorrect items -- agreement means
                # both gemini judge_score and phi4 judge_verdict concur.
                gem = r.get("judge_score", "")
                phi4_row = phi4_by_id.get(r["id"])
                if gem == "" or phi4_row is None:
                    continue
                phi4_v = phi4_row["judge_verdict"]
                if gem != phi4_v:
                    continue  # disagreement -- already in the 174-row sheet
                base["gemini_verdict"] = gem
                base["phi4_verdict"] = phi4_v
                add(f"{tshort}_strict_incorrect_judges_agree", base)
            else:
                # Gemini never judged strict-correct items; phi4 covers all of them.
                phi4_row = phi4_by_id.get(r["id"])
                if phi4_row is None or phi4_row["judge_verdict"] != "1":
                    continue  # phi4 disputes it (a strict-FP candidate) -- not an agreement case
                base["gemini_verdict"] = ""
                base["phi4_verdict"] = "1"
                add(f"{tshort}_strict_correct_phi4_agrees", base)

    # CON: no judge at all -- direct strict-correct / strict-incorrect spot-check pools.
    for path in sorted((ROOT / "evaluation" / "results").glob("*.csv")):
        key = path.stem.replace("_results", "")
        if key not in MODEL_KEY_TO_LABEL:
            continue
        label = MODEL_KEY_TO_LABEL[key]
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if r["task_type"] != "contradiction_detection":
                continue
            base = {
                "model": label, "item_id": r["id"], "task_type": r["task_type"],
                "strict_correct": r["correct"], "gemini_verdict": "", "phi4_verdict": "",
                "question": r["question"], "reference_answer": r["ref_answer"],
                "model_prediction": r["prediction"],
            }
            stratum = "CON_strict_incorrect_no_judge" if r["correct"] == "0" else "CON_strict_correct_no_judge"
            add(stratum, base)

    return pools


def main():
    existing = load_existing()
    existing_keys = {(r["model"], r["item_id"]) for r in existing}

    pools = build_pools()
    rng = random.Random(SEED)
    new_rows = []
    print(f"{'stratum':<38}{'pool size':>10}{'sampled':>9}")
    for stratum in sorted(pools):
        pool = [p for p in pools[stratum] if (p["model"], p["item_id"]) not in existing_keys]
        overlap = len(pools[stratum]) - len(pool)
        assert overlap == 0, f"{stratum}: {overlap} items unexpectedly already in the 174-row sheet"
        k = min(N_PER_STRATUM, len(pool))
        sample = rng.sample(pool, k)
        for r in sample:
            r["sample_group"] = "control_agreement"
            r["stratum"] = stratum
            for f in FIELDNAMES:
                r.setdefault(f, "")
        new_rows.extend(sample)
        print(f"{stratum:<38}{len(pool):>10}{k:>9}")

    combined = existing + new_rows
    with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(combined)

    print(f"\nexisting (disagreement) rows: {len(existing)}")
    print(f"new (control_agreement) rows: {len(new_rows)}")
    print(f"total: {len(combined)}")


if __name__ == "__main__":
    main()
