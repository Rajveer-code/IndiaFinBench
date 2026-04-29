"""
generate_iaa_expansion.py
--------------------------
Generates the 60-item IAA expansion sheet for IndiaFinBench.

Samples 60 NEW items (not already in the 60-item IAA) stratified by task type:
  REG=26, NUM=14, CON=9, TMP=11  (matches benchmark distribution)

Outputs:
  annotation/inter_annotator/iaa_expansion_ANNOTATOR_SHEET.csv  <- send to annotator
  annotation/inter_annotator/iaa_expansion_REFERENCE.csv        <- keep private

Usage:
  python scripts/generate_iaa_expansion.py
"""

import csv
import json
import random
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

QA_PATH        = "annotation/raw_qa/indiafinbench_qa_combined_406.json"
EXISTING_IAA   = "annotation/inter_annotator/kappa_report.csv"
SHEET_PATH     = "annotation/inter_annotator/iaa_expansion_ANNOTATOR_SHEET.csv"
REFERENCE_PATH = "annotation/inter_annotator/iaa_expansion_REFERENCE.csv"

SEED = 42

# Target counts per task type (matches benchmark distribution ~43/23/15/19%)
TARGETS = {
    "regulatory_interpretation": 26,
    "numerical_reasoning":       14,
    "contradiction_detection":    9,
    "temporal_reasoning":        11,
}

SHORT = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}


def load_existing_ids(path: str) -> set:
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.add(row["id"])
    return ids


def build_context(item: dict) -> str:
    if item["task_type"] == "contradiction_detection":
        a = item.get("context_a", "")
        b = item.get("context_b", "")
        return f"Passage A:\n{a}\n\nPassage B:\n{b}"
    return item.get("context", "")


def main():
    with open(QA_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    existing_ids = load_existing_ids(EXISTING_IAA)
    print(f"  Total benchmark items  : {len(all_items)}")
    print(f"  Existing IAA item IDs  : {len(existing_ids)}")

    available = [item for item in all_items if item["id"] not in existing_ids]
    print(f"  Available for sampling : {len(available)}")

    # Bucket by task type
    buckets: dict[str, list] = {t: [] for t in TARGETS}
    for item in available:
        t = item["task_type"]
        if t in buckets:
            buckets[t].append(item)

    print()
    print(f"  {'Task':<32}  {'Available':>9}  {'Target':>6}")
    print(f"  {'─'*32}  {'─'*9}  {'─'*6}")
    for t, n in TARGETS.items():
        avail = len(buckets[t])
        ok = "OK" if avail >= n else f"WARN: only {avail} available!"
        print(f"  {t:<32}  {avail:>9}  {n:>6}  {ok}")
    print()

    rng = random.Random(SEED)
    selected = []
    for t, n in TARGETS.items():
        pool = buckets[t]
        rng.shuffle(pool)
        chosen = pool[:n]
        selected.extend(chosen)
        print(f"  Sampled {len(chosen):>2} {SHORT[t]} items (seed={SEED})")

    # Shuffle combined list with same seed for random presentation order
    rng.shuffle(selected)

    # Write annotator sheet (blank YOUR_ANSWER column)
    os.makedirs(os.path.dirname(SHEET_PATH), exist_ok=True)
    with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "task_type", "context", "question", "YOUR_ANSWER"])
        w.writeheader()
        for item in selected:
            w.writerow({
                "item_id":     item["id"],
                "task_type":   item["task_type"],
                "context":     build_context(item),
                "question":    item["question"],
                "YOUR_ANSWER": "",
            })

    # Write reference CSV (private — for scoring only)
    with open(REFERENCE_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "task_type", "reference_answer"])
        w.writeheader()
        for item in selected:
            w.writerow({
                "item_id":          item["id"],
                "task_type":        item["task_type"],
                "reference_answer": item["answer"],
            })

    print()
    print(f"  ── Summary ──────────────────────────────")
    print(f"  {'Task':<32}  {'n':>4}")
    print(f"  {'─'*32}  {'─'*4}")
    from collections import Counter
    counts = Counter(item["task_type"] for item in selected)
    total = 0
    for t in TARGETS:
        n = counts.get(t, 0)
        total += n
        print(f"  {t:<32}  {n:>4}")
    print(f"  {'─'*32}  {'─'*4}")
    print(f"  {'TOTAL':<32}  {total:>4}")
    print()
    print(f"  Annotator sheet  → {SHEET_PATH}")
    print(f"  Reference (keep private) → {REFERENCE_PATH}")
    print()
    print("  Next: send ANNOTATOR_SHEET.csv to your second annotator.")
    print("  When returned, run: python scripts/score_iaa_expansion.py \\")
    print("      --returned_sheet <path_to_filled_sheet>")


if __name__ == "__main__":
    main()
