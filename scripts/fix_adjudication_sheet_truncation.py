"""One-time patch: adjudication_sheet.csv's question/reference_answer/model_prediction
came from evaluation/results*/*.csv, which truncates those fields to 80/100/200 chars
at logging time (scripts/evaluate.py lines 394-396) -- a display bug, not a scoring bug
(score_answer() runs on full text before truncation, confirmed by reading the source).

question and reference_answer are fully recoverable from the dataset JSON (never
truncated there). model_prediction is NOT recoverable for rows where it was actually
cut at 200 chars -- the raw API response was never saved anywhere else. Those rows get
a prediction_truncated=TRUE flag instead of silently pretending the text is complete.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SHEET_PATH = ROOT / "annotation" / "judge_audit" / "adjudication_sheet.csv"

dataset = json.loads((ROOT / "annotation/raw_qa/indiafinbench_qa_combined_406.json").read_text(encoding="utf-8"))
if isinstance(dataset, dict):
    dataset = dataset.get("items", dataset.get("data", list(dataset.values())[0]))
full_by_id = {item["id"]: item for item in dataset}

with open(SHEET_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames)
    rows = list(reader)

if "prediction_truncated" not in fieldnames:
    fieldnames.insert(fieldnames.index("model_prediction") + 1, "prediction_truncated")

n_q_fixed = n_ref_fixed = n_pred_flagged = 0
for r in rows:
    was_pred_truncated = len(r["model_prediction"]) == 200
    item = full_by_id.get(r["item_id"])
    if item:
        if r["question"] != item["question"]:
            r["question"] = item["question"]
            n_q_fixed += 1
        if r["reference_answer"] != item["answer"]:
            r["reference_answer"] = item["answer"]
            n_ref_fixed += 1
    r["prediction_truncated"] = "TRUE" if was_pred_truncated else "FALSE"
    if was_pred_truncated:
        n_pred_flagged += 1

with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"questions restored to full text: {n_q_fixed}")
print(f"reference answers restored to full text: {n_ref_fixed}")
print(f"predictions flagged as truncated (unrecoverable, not fixed): {n_pred_flagged}")
