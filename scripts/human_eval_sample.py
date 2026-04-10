"""
human_eval_sample.py
Generates 30-item human evaluation sheet.
Run from IndiaFinBench/ root.
"""
import json, csv, random
from pathlib import Path

QA_PATH    = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
OUT_DIR    = Path("annotation/human_eval")
SHEET_PATH = OUT_DIR / "human_eval_answer_sheet(AutoRecovered)"
REF_PATH   = OUT_DIR / "human_eval_with_refs.csv"
SEED       = 42
random.seed(SEED)

with open(QA_PATH, encoding="utf-8") as f:
    data = json.load(f)

by_diff = {"easy": [], "medium": [], "hard": []}
for item in data:
    d = item.get("difficulty", "medium").lower()
    if d in by_diff: by_diff[d].append(item)

sampled = []
for diff, items in by_diff.items():
    sampled.extend(random.sample(items, min(10, len(items))))

from collections import Counter
print(f"Sampled {len(sampled)} items")
print("  Difficulty:", Counter(i["difficulty"] for i in sampled))
print("  Task type:", Counter(i["task_type"] for i in sampled))

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sheet WITH reference answers (check your score after)
with open(REF_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","task_type","difficulty","context","question","reference_answer"])
    w.writeheader()
    for item in sampled:
        ctx = item.get("context", "")
        if item.get("task_type") == "contradiction_detection":
            ctx = f"Passage A:\n{item.get('context_a','')}\n\nPassage B:\n{item.get('context_b','')}"
        w.writerow({"id": item["id"], "task_type": item["task_type"],
                    "difficulty": item["difficulty"], "context": ctx,
                    "question": item["question"], "reference_answer": item["answer"]})

# Answer sheet WITHOUT reference answers (fill this in yourself)
with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","task_type","difficulty","context","question","your_answer"])
    w.writeheader()
    for item in sampled:
        ctx = item.get("context", "")
        if item.get("task_type") == "contradiction_detection":
            ctx = f"Passage A:\n{item.get('context_a','')}\n\nPassage B:\n{item.get('context_b','')}"
        w.writerow({"id": item["id"], "task_type": item["task_type"],
                    "difficulty": item["difficulty"], "context": ctx,
                    "question": item["question"], "your_answer": ""})

print(f"\nFiles created:")
print(f"  {SHEET_PATH}  ← OPEN THIS. Fill in 'your_answer' column for all 30 rows.")
print(f"  {REF_PATH}   ← Check your answers against this AFTER filling.")