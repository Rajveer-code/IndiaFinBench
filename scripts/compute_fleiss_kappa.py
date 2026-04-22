"""
Compute Fleiss' kappa from 3 annotators on the 90-item validation sheet.

Annotator 1: You (primary author)  — annotator1_completed.csv
Annotator 2: Claude (AI validator) — annotator2_completed.csv
Annotator 3: Gemini (AI validator) — annotator3_completed.csv

Run: python scripts/compute_fleiss_kappa.py
"""
import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MA_DIR    = REPO_ROOT / "annotation/multi_annotator"

ANNOTATOR_FILES = {
    "ann1_author":  MA_DIR / "annotator1_completed.csv",
    "ann2_claude":  MA_DIR / "annotator2_completed.csv",
    "ann3_gemini":  MA_DIR / "annotator3_completed.csv",
}


def parse_yn(val) -> int:
    v = str(val).strip().lower()
    if v in ('yes', 'y', '1'):  return 1
    if v in ('no',  'n', '0'):  return 0
    raise ValueError(f"Cannot parse: {val!r}")


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa. ratings[i,j] = number of raters assigning category j to item i."""
    N, k   = ratings.shape
    n      = int(ratings[0].sum())
    p_j    = ratings.sum(axis=0) / (N * n)
    P_i    = ((ratings ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar  = P_i.mean()
    P_e    = (p_j ** 2).sum()
    return round((P_bar - P_e) / (1 - P_e), 4) if P_e < 1 else 1.0


def interpret(k: float) -> str:
    if k < 0:    return "poor"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    # Load each annotator
    dfs = {}
    for name, path in ANNOTATOR_FILES.items():
        if not path.exists():
            print(f"  MISSING: {path.name}  (skipping {name})")
            continue
        df = pd.read_csv(path)
        df['label'] = df['is_correct__YES_or_NO'].apply(parse_yn)
        dfs[name] = df.set_index('id')['label']
        yes_n = df['label'].sum()
        print(f"  Loaded {name}: {len(df)} rows  YES={yes_n}  NO={len(df)-yes_n}")

    if len(dfs) < 2:
        print("Need at least 2 annotator files. See plan for filling them.")
        return

    # Merge on common item IDs
    merged = pd.DataFrame(dfs).dropna()
    print(f"\nItems with all {len(dfs)} annotator responses: {len(merged)}")

    # Load task types from reference
    ref = pd.read_csv(MA_DIR / "annotation_reference.csv").set_index('id')
    merged = merged.join(ref[['task_type']], how='left')

    ann_cols = [c for c in merged.columns if c.startswith('ann')]

    # ── Overall Fleiss' κ ────────────────────────────────────────────────────
    ratings = np.zeros((len(merged), 2), dtype=float)
    for col in ann_cols:
        ratings[:, 1] += merged[col].values.astype(float)
    ratings[:, 0] = len(ann_cols) - ratings[:, 1]

    overall_k     = fleiss_kappa(ratings)
    overall_agree = merged[ann_cols].apply(
        lambda r: r.nunique() == 1, axis=1).mean() * 100

    print(f"\n{'='*55}")
    print(f"FLEISS' KAPPA  ({len(ann_cols)} annotators, {len(merged)} items)")
    print(f"{'='*55}")
    print(f"Overall:  κ = {overall_k:.4f}  ({overall_agree:.1f}% full agreement)")
    print(f"Interpretation: {interpret(overall_k)}")

    rows = []
    for task in ['regulatory_interpretation', 'numerical_reasoning',
                 'contradiction_detection', 'temporal_reasoning']:
        t_df = merged[merged['task_type'] == task] if 'task_type' in merged.columns \
               else merged
        if len(t_df) == 0:
            continue

        t_ratings = np.zeros((len(t_df), 2), dtype=float)
        for col in ann_cols:
            t_ratings[:, 1] += t_df[col].values.astype(float)
        t_ratings[:, 0] = len(ann_cols) - t_ratings[:, 1]

        t_k     = fleiss_kappa(t_ratings)
        t_agree = t_df[ann_cols].apply(lambda r: r.nunique() == 1, axis=1).mean() * 100

        short = {'regulatory_interpretation':'REG','numerical_reasoning':'NUM',
                 'contradiction_detection':'CON','temporal_reasoning':'TMP'}.get(task, task)
        print(f"  {short:3s}  n={len(t_df):2d}  κ={t_k:.4f}  agree={t_agree:.0f}%  "
              f"({interpret(t_k)})")

        rows.append({'task_type': task, 'n_items': len(t_df),
                     'agreement_pct': round(t_agree, 1),
                     'fleiss_kappa': t_k,
                     'interpretation': interpret(t_k)})

    rows.append({'task_type': 'OVERALL', 'n_items': len(merged),
                 'agreement_pct': round(overall_agree, 1),
                 'fleiss_kappa': overall_k,
                 'interpretation': interpret(overall_k)})

    result_df = pd.DataFrame(rows)
    out_path = MA_DIR / "fleiss_kappa_report.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Show disagreement items (where annotators differ)
    disagree = merged[merged[ann_cols].apply(lambda r: r.nunique() > 1, axis=1)]
    if len(disagree) > 0:
        print(f"\nItems with disagreement ({len(disagree)}):")
        print(disagree[ann_cols + ['task_type']].to_string())

    return result_df


if __name__ == "__main__":
    main()
