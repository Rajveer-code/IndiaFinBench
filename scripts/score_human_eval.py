"""
score_human_eval.py  
Scores your human eval answers and prints paper-ready numbers.
Run AFTER filling human_eval_answer_sheet.csv.
"""
import csv, re, json
from pathlib import Path
from collections import defaultdict

try:
    from rapidfuzz import fuzz
except ImportError:
    raise ImportError("pip install rapidfuzz")

SHEET_PATH = Path(r"D:\Projects\IndiaFinBench\annotation\human_eval\human_annotator_answer_sheet_filled.csv")
QA_PATH    = Path(r"D:\Projects\IndiaFinBench\annotation\raw_qa\indiafinbench_qa_combined_406.json")

with open(QA_PATH, encoding="utf-8") as f:
    qa_data = json.load(f)
ref_map = {item["id"]: item for item in qa_data}

with open(SHEET_PATH, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

def normalise(text):
    if not text: return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def score_item(ref, pred, task_type):
    rn, pn = normalise(ref), normalise(pred)
    if rn == pn: return 1, "exact"
    if task_type == "contradiction_detection":
        ry = "yes" if rn.startswith("yes") else ("no" if rn.startswith("no") else "")
        py = "yes" if pn.startswith("yes") else ("no" if pn.startswith("no") else "")
        if ry and py: return (1,"yn") if ry==py else (0,"yn_miss")
    rn2 = set(re.findall(r"\d+[\d,]*\.?\d*", rn))
    pn2 = set(re.findall(r"\d+[\d,]*\.?\d*", pn))
    if rn2 and pn2 and rn2 == pn2: return 1, "numerical"
    if fuzz.token_set_ratio(rn, pn) / 100.0 >= 0.72: return 1, "fuzzy"
    return 0, "incorrect"

by_task = defaultdict(lambda: {"c":0,"t":0})
total_c = 0

for row in rows:
    iid      = row["id"]
    your_ans = row.get("your_answer","").strip()
    ref_item = ref_map.get(iid)
    if not ref_item: continue
    if not your_ans:
        print(f"  WARNING: No answer for {iid}"); continue
    correct, _ = score_item(ref_item["answer"], your_ans, row["task_type"])
    by_task[row["task_type"]]["c"] += correct
    by_task[row["task_type"]]["t"] += 1
    total_c += correct

total_n = sum(v["t"] for v in by_task.values())
overall = total_c / total_n * 100 if total_n else 0

TASK_MAP = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}

print("\n" + "="*55)
print("  HUMAN EXPERT EVALUATION RESULTS")
print("  Copy these into the paper")
print("="*55)
print(f"  Overall: {overall:.1f}%  ({total_c}/{total_n})")

task_scores = {}
for task_full, ts in TASK_MAP.items():
    d = by_task.get(task_full, {"c":0,"t":0})
    if d["t"] > 0:
        pct = d["c"] / d["t"] * 100
        task_scores[ts] = pct
        print(f"  {ts}: {pct:.1f}%  ({d['c']}/{d['t']})")

gap = overall - 91.3
print(f"\n  Gap vs Claude 3 Haiku (91.3%): {gap:+.1f}pp")
if gap > 3:
    resolution = "substantial room for model improvement"
elif gap < -3:
    resolution = "near-human or above-human parity on this subset"
else:
    resolution = "near-human parity on this subset"

print(f"\n  PASTE INTO PAPER:")
print(f"  Table 1 Human Expert row:")
for ts in ["REG","NUM","CON","TMP"]:
    print(f"    {ts}: {task_scores.get(ts, 0):.1f}%")
print(f"    Overall: {overall:.1f}%")
print(f"\n  Section 5.1 conditional: '{resolution}'")
print("="*55)