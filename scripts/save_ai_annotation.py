"""
Save the Claude annotation responses (from claude.ai) as annotator2_completed.csv.
Parses the "Row N: YES/NO | reason" format.

Run once, then run compute_fleiss_kappa.py.
"""
import sys, re
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MA_DIR = REPO_ROOT / "annotation/multi_annotator"

# ── Claude's annotation responses (pasted here) ─────────────────────────────
CLAUDE_RESPONSES = """
Row 1: YES |
Row 2: YES |
Row 3: YES |
Row 4: YES |
Row 5: YES |
Row 6: YES |
Row 7: YES |
Row 8: YES |
Row 9: YES |
Row 10: YES |
Row 11: YES |
Row 12: YES |
Row 13: YES |
Row 14: YES |
Row 15: YES |
Row 16: YES |
Row 17: YES |
Row 18: YES |
Row 19: YES |
Row 20: YES |
Row 21: YES |
Row 22: YES |
Row 23: YES |
Row 24: YES |
Row 25: YES |
Row 26: YES |
Row 27: YES |
Row 28: NO | Context is truncated mid-sentence
Row 29: YES |
Row 30: YES |
Row 31: YES |
Row 32: YES |
Row 33: YES |
Row 34: YES |
Row 35: YES |
Row 36: YES |
Row 37: YES |
Row 38: YES |
Row 39: NO | The 750 crore debt figure requires assumptions not explicit in context
Row 40: YES |
Row 41: YES |
Row 42: YES |
Row 43: YES |
Row 44: YES |
Row 45: YES |
Row 46: YES |
Row 47: NO | Passage A permits unlisted equity; answer No is incorrect for Passage A
Row 48: YES |
Row 49: YES |
Row 50: YES |
Row 51: YES |
Row 52: YES |
Row 53: YES |
Row 54: YES |
Row 55: YES |
Row 56: NO | Passage A has a proviso raising threshold to 15% for some entities; No is incorrect
Row 57: YES |
Row 58: YES |
Row 59: YES |
Row 60: YES |
Row 61: YES |
Row 62: YES |
Row 63: YES |
Row 64: YES |
Row 65: YES |
Row 66: YES |
Row 67: YES |
Row 68: YES |
Row 69: YES |
Row 70: YES |
Row 71: YES |
Row 72: YES |
Row 73: YES |
Row 74: YES |
Row 75: YES |
Row 76: YES |
Row 77: YES |
Row 78: YES |
Row 79: YES |
Row 80: YES |
Row 81: YES |
Row 82: YES |
Row 83: YES |
Row 84: YES |
Row 85: YES |
Row 86: YES |
Row 87: YES |
Row 88: YES |
Row 89: YES |
Row 90: YES |
"""


def parse_responses(text: str) -> dict[int, dict]:
    """Parse 'Row N: YES/NO | reason' format into {row_number: {label, note}}."""
    results = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'Row\s+(\d+):\s*(YES|NO)\s*\|?\s*(.*)', line, re.IGNORECASE)
        if m:
            row_num = int(m.group(1))
            label = m.group(2).upper()
            note = m.group(3).strip()
            results[row_num] = {'label': label, 'note': note}
    return results


def main():
    sheet = pd.read_csv(MA_DIR / "annotation_sheet_90items.csv")
    print(f"Annotation sheet: {len(sheet)} rows")

    parsed = parse_responses(CLAUDE_RESPONSES)
    print(f"Parsed {len(parsed)} responses from Claude")

    # Build completed CSV
    completed = sheet.copy()
    completed['is_correct__YES_or_NO'] = ''
    completed['notes_if_NO'] = ''

    for _, row in completed.iterrows():
        rn = int(row['row_number'])
        if rn in parsed:
            completed.at[row.name, 'is_correct__YES_or_NO'] = parsed[rn]['label']
            completed.at[row.name, 'notes_if_NO'] = parsed[rn]['note']

    filled = (completed['is_correct__YES_or_NO'] != '').sum()
    print(f"Filled: {filled}/{len(completed)} rows")

    yes_count = (completed['is_correct__YES_or_NO'] == 'YES').sum()
    no_count  = (completed['is_correct__YES_or_NO'] == 'NO').sum()
    print(f"YES: {yes_count}  NO: {no_count}")

    out_path = MA_DIR / "annotator2_completed.csv"
    completed.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    # Show the 4 NOs
    nos = completed[completed['is_correct__YES_or_NO'] == 'NO']
    print("\nItems marked NO (genuine dataset issues to fix):")
    for _, r in nos.iterrows():
        print(f"  Row {r['row_number']} ({r['task_type']}, {r['difficulty']}): {r['notes_if_NO'][:80]}")


if __name__ == "__main__":
    main()
