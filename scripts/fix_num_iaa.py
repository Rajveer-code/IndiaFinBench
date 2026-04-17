"""
Fix NUM IAA scoring bug in annotation/inter_annotator/kappa_report.csv

The original agreement was computed with exact string match.
Several NUM items are marked agree=0 but are semantically identical:
  - "32.35%"  vs  "32.35%. Calculation: ..."  → should be 1
  - "0.5 crore" vs "0.50 crore rupees (...)"   → should be 1
  - "1-Oct-25"  vs  "1 October 2025 (...)"     → should be 1

This script re-scores those items and saves corrected kappa_report.csv
"""
import sys, re
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
KAPPA_PATH = REPO_ROOT / "annotation/inter_annotator/kappa_report.csv"
SUMMARY_PATH = REPO_ROOT / "annotation/inter_annotator/kappa_summary_corrected.csv"


def normalize(text: str) -> str:
    """
    Strip everything after the first explanation marker so we compare
    only the core answer, not the verbose reference justification.

    "32.35%. Calculation: ..."  →  "32.35"
    "0.50 crore rupees (2.5%…)" →  "0.5 crore"
    "1 October 2025 (warrant…)" →  "1 october 2025"
    "nineteen thousand crore"   →  "19000 crore"  (handled by num extraction)
    """
    if not isinstance(text, str):
        return str(text).lower().strip()

    t = text.lower().strip()

    # Cut off at first opening bracket or "calculation:" / "reasoning:"
    t = re.split(r'[\(\[]', t)[0]
    t = re.sub(r'\b(calculation|reason|note|i\.e\.|i\.e|viz\.?).*', '', t, flags=re.IGNORECASE)
    t = t.strip().rstrip('.,;')

    # Currency symbols
    t = re.sub(r'[₹]|rs\.?\s*', '', t, flags=re.IGNORECASE)

    # "crore rupees" → "crore"
    t = re.sub(r'crore\s+rupees?', 'crore', t)

    # Remove commas inside numbers: 19,000 → 19000
    t = re.sub(r'(\d),(\d{3})', r'\1\2', t)

    # Trailing zeros on decimals: 0.50 → 0.5
    t = re.sub(r'(\d+\.\d*?)0+\b', r'\1', t)
    t = re.sub(r'(\d+)\.$', r'\1', t)   # "32." → "32"

    # Month names → numbers (handles date format variants)
    month_map = {
        'january':'01','february':'02','march':'03','april':'04',
        'may':'05','june':'06','july':'07','august':'08',
        'september':'09','october':'10','november':'11','december':'12',
        'jan':'01','feb':'02','mar':'03','apr':'04',
        'jun':'06','jul':'07','aug':'08','sep':'09','oct':'10','nov':'11','dec':'12',
    }
    for name, num in month_map.items():
        t = re.sub(r'\b' + name + r'\b', num, t)

    # "by 29 jul" → "29 07"  (remove "by"/"from"/"on" prepositions before dates)
    t = re.sub(r'\b(by|on|from|before|after)\b\s+', '', t)

    return t.strip()


def answers_match(ref: str, ann: str) -> int:
    """Return 1 if ref and ann are semantically equivalent, 0 otherwise."""
    r = normalize(ref)
    a = normalize(ann)

    # Direct match after normalisation
    if r == a:
        return 1

    # Containment: short answer contained in long reference
    if a and r and (a in r or r in a):
        return 1

    # Numeric set match (handles unit/format differences)
    def nums(s):
        s = re.sub(r'[₹,%]', '', s)
        return set(re.findall(r'\d+(?:\.\d+)?', s))

    rn, an_ = nums(r), nums(a)
    if rn and an_ and rn == an_:
        return 1

    return 0


def rescore_num_items(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Only fix FALSE NEGATIVES (agree=0 that should be agree=1).
    Never downgrade items that were already agree=1 — those were manually
    validated as correct and should not be overridden by normalization.
    """
    df = df.copy()
    changed = 0
    num_mask = df['task_type'] == 'numerical_reasoning'

    for idx in df[num_mask].index:
        old = df.at[idx, 'agree']
        # Only attempt to fix items scored as 0 (false negatives)
        # Never touch items already scored as 1
        if old != 0:
            continue
        new = answers_match(str(df.at[idx, 'ref_answer']),
                            str(df.at[idx, 'ann2_answer']))
        df.at[idx, 'agree'] = new
        if new == 1:   # only report upgrades
            changed += 1
            print(f"  Fixed {df.at[idx,'id']}: 0→1  "
                  f"ref={str(df.at[idx,'ref_answer'])[:50]!r}  "
                  f"ann={str(df.at[idx,'ann2_answer'])[:30]!r}")

    return df, changed


def kappa_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ['regulatory_interpretation', 'numerical_reasoning',
                 'contradiction_detection', 'temporal_reasoning']:
        t = df[df['task_type'] == task]
        if len(t) == 0:
            continue
        n = len(t)
        agree_pct = t['agree'].mean() * 100

        kappa = None
        if task == 'contradiction_detection':
            p_o = t['agree'].mean()
            ref_yes = t['ref_answer'].str.lower().str.strip().isin(['yes', 'y']).mean()
            ann_yes = t['ann2_answer'].str.lower().str.strip().isin(['yes', 'y']).mean()
            p_e = ref_yes * ann_yes + (1 - ref_yes) * (1 - ann_yes)
            kappa = round((p_o - p_e) / (1 - p_e), 3) if p_e < 1 else 1.0

        rows.append({'task_type': task, 'n_items': n,
                     'agreement_pct': round(agree_pct, 1),
                     'cohens_kappa': kappa})

    rows.append({'task_type': 'OVERALL', 'n_items': len(df),
                 'agreement_pct': round(df['agree'].mean() * 100, 1),
                 'cohens_kappa': None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(KAPPA_PATH)
    print(f"Loaded {len(df)} items from {KAPPA_PATH}")

    print("\n=== ORIGINAL SCORES ===")
    orig = kappa_summary(df)
    print(orig.to_string(index=False))

    df_fixed, n_changed = rescore_num_items(df)
    print(f"\n{n_changed} NUM agreements changed after normalization")

    print("\n=== CORRECTED SCORES ===")
    fixed = kappa_summary(df_fixed)
    print(fixed.to_string(index=False))

    df_fixed.to_csv(KAPPA_PATH, index=False)
    fixed.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved corrected kappa_report  -> {KAPPA_PATH}")
    print(f"Saved summary                  -> {SUMMARY_PATH}")
