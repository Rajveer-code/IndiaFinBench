"""
compute_kappa.py
----------------
Computes Cohen's Kappa between two annotator response files.

Usage:
    python scripts/compute_kappa.py \
        --a annotation/inter_annotator/annotator1.csv \
        --b annotation/inter_annotator/annotator2.csv
"""
# Full implementation added in Phase 2
import argparse, pandas as pd
from sklearn.metrics import cohen_kappa_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    args = parser.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    merged = df_a.merge(df_b, on="id", suffixes=("_a","_b"))
    kappa  = cohen_kappa_score(merged["answer_a"], merged["answer_b"])
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Items compared: {len(merged)}")

if __name__ == "__main__":
    main()
