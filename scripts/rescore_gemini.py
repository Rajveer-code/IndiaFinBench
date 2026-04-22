"""
rescore_gemini.py
-----------------
Re-score gemini25_pro_results.csv using the correct scoring logic.
The predictions in the CSV are correct — only the `correct` column needs fixing.

Applies:
  - Exact match
  - Fuzzy token match (threshold=72)
  - Numeric extraction match
  - Regex YES/NO for contradiction_detection (fixes the "Yes, explanation..." bug)

Run: python scripts/rescore_gemini.py
"""
import re, sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
import pandas as pd
from pathlib import Path

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("Run: pip install rapidfuzz")

RESULTS_DIR = Path(__file__).parent.parent / "evaluation/results"
CSV_PATH    = RESULTS_DIR / "gemini25_pro_results.csv"


def score(prediction: str, reference: str, task_type: str) -> int:
    pred = str(prediction).lower().strip()
    ref  = str(reference).lower().strip()

    if pred == ref:
        return 1
    if _rf.token_set_ratio(pred, ref) >= 72:
        return 1

    def extract_nums(s):
        s = re.sub(r'[₹,]', '', s)
        return set(re.findall(r'\d+(?:\.\d+)?', s))
    pn, rn = extract_nums(pred), extract_nums(ref)
    if pn and rn and pn == rn:
        return 1

    # Contradiction: regex YES/NO — handles "Yes, Passage A..." "No. Both..." etc.
    if 'contradiction' in str(task_type).lower():
        p0 = re.search(r'\b(yes|no)\b', pred)
        r0 = re.search(r'\b(yes|no)\b', ref)
        if p0 and r0:
            return 1 if p0.group() == r0.group() else 0
        return 0

    return 0


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} items from {CSV_PATH.name}")

    print("\n=== BEFORE re-score ===")
    for task, grp in df.groupby('task_type'):
        short = {'regulatory_interpretation':'REG','numerical_reasoning':'NUM',
                 'contradiction_detection':'CON','temporal_reasoning':'TMP'}.get(task, task)
        print(f"  {short}: {grp['correct'].mean()*100:.1f}%  (n={len(grp)}, correct={grp['correct'].sum()})")
    print(f"  OVERALL: {df['correct'].mean()*100:.1f}%")

    # Re-score every row
    df['correct'] = df.apply(
        lambda r: score(str(r['prediction']), str(r['ref_answer']), str(r['task_type'])),
        axis=1
    )

    print("\n=== AFTER re-score ===")
    for task, grp in df.groupby('task_type'):
        short = {'regulatory_interpretation':'REG','numerical_reasoning':'NUM',
                 'contradiction_detection':'CON','temporal_reasoning':'TMP'}.get(task, task)
        print(f"  {short}: {grp['correct'].mean()*100:.1f}%  (n={len(grp)}, correct={grp['correct'].sum()})")
    print(f"  OVERALL: {df['correct'].mean()*100:.1f}%")

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved corrected scores -> {CSV_PATH}")


if __name__ == "__main__":
    main()
