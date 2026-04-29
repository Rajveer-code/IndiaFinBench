"""Save Gemini 2.5 Pro annotation responses as annotator3_completed.csv."""
import sys, re
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
import pandas as pd
from pathlib import Path

MA_DIR = Path(__file__).parent.parent / "annotation/multi_annotator"

GEMINI_RESPONSES = """
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
Row 15: NO | Reference answer includes previous five years not explicit in context
Row 16: YES |
Row 17: YES |
Row 18: YES |
Row 19: YES |
Row 20: YES |
Row 21: NO | Reference answer includes details about SEBI directions cut off in context
Row 22: YES |
Row 23: YES |
Row 24: YES |
Row 25: YES |
Row 26: NO | Reference answer includes full text of fourth condition cut off in context
Row 27: YES |
Row 28: YES |
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
Row 39: YES |
Row 40: YES |
Row 41: YES |
Row 42: YES |
Row 43: YES |
Row 44: YES |
Row 45: YES |
Row 46: YES |
Row 47: YES |
Row 48: YES |
Row 49: NO | Passage B is cut off and minimum investment threshold cannot be verified
Row 50: YES |
Row 51: YES |
Row 52: YES |
Row 53: YES |
Row 54: YES |
Row 55: YES |
Row 56: YES |
Row 57: YES |
Row 58: YES |
Row 59: YES |
Row 60: YES |
Row 61: YES |
Row 62: YES |
Row 63: YES |
Row 64: NO | Reference answer includes 2025 amendment info not in context
Row 65: YES |
Row 66: NO | Reference answer mentions Companies Act section 247 not in context
Row 67: YES |
Row 68: YES |
Row 69: NO | Reference answer states specific reduced amount Rs 25 crore not in context
Row 70: YES |
Row 71: YES |
Row 72: NO | Reference answer details Chapter IVA regulations not specified in context
Row 73: YES |
Row 74: YES |
Row 75: YES |
Row 76: YES |
Row 77: YES |
Row 78: YES |
Row 79: NO | Reference answer includes Regulation 9(e) details not in context
Row 80: YES |
Row 81: YES |
Row 82: YES |
Row 83: YES |
Row 84: NO | Reference answer includes housing finance company details not in context
Row 85: NO | Reference answer includes two-payment structure details not in context
Row 86: NO | Reference answer includes commentary on same person fulfilling requirements
Row 87: YES |
Row 88: YES |
Row 89: YES |
Row 90: NO | Reference answer mentions LODR 2015 subsumption not stated in context
"""

def parse(text):
    results = {}
    for line in text.strip().splitlines():
        m = re.match(r'Row\s+(\d+):\s*(YES|NO)\s*\|?\s*(.*)', line.strip(), re.IGNORECASE)
        if m:
            results[int(m.group(1))] = {'label': m.group(2).upper(), 'note': m.group(3).strip()}
    return results

sheet = pd.read_csv(MA_DIR / "annotation_sheet_90items.csv")
parsed = parse(GEMINI_RESPONSES)

completed = sheet.copy()
completed['is_correct__YES_or_NO'] = ''
completed['notes_if_NO'] = ''

for _, row in completed.iterrows():
    rn = int(row['row_number'])
    if rn in parsed:
        completed.at[row.name, 'is_correct__YES_or_NO'] = parsed[rn]['label']
        completed.at[row.name, 'notes_if_NO'] = parsed[rn]['note']

yes_n = (completed['is_correct__YES_or_NO'] == 'YES').sum()
no_n  = (completed['is_correct__YES_or_NO'] == 'NO').sum()
print(f"Gemini annotation: YES={yes_n}  NO={no_n}")
completed.to_csv(MA_DIR / "annotator3_completed.csv", index=False)
print(f"Saved -> {MA_DIR}/annotator3_completed.csv")

# Show NOs
print("\nItems Gemini flagged NO:")
for _, r in completed[completed['is_correct__YES_or_NO']=='NO'].iterrows():
    print(f"  Row {r['row_number']} ({r['task_type']}, {r['difficulty']}): {r['notes_if_NO'][:70]}")
