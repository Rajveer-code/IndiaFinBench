"""
item_discrimination_exact.py
------------------------------
Cleanup pass, 2026-09-03: draft_03's discriminative-coverage section (Section
6, tab:effsize) claimed the 213-item "ceiling" band (p >= 0.9, i.e. k in
{11,12} of 12 models correct) "carries almost no information about the
relative ordering of these systems." That is only true for k=12 (0 of the 66
model pairs disagree on that item). For k=11, exactly one model differs from
the other eleven, so the item still separates 11 of 66 pairs -- real signal,
not noise. Conflating k=11 with k=12 under one "ceiling" label overclaims.
This script computes the exact pairwise-disagreement measure D(k) = k(12-k)
(the number of model pairs an item with k correct responses distinguishes)
alongside the p-based bands already in evaluation/threshold_sensitivity.json,
using the same frozen correctness matrix everyone else uses (load_correctness_
matrix() -> evaluation/results/*.csv's original `correct` column, unaffected
by the judge/truncation-fix work since that was a scoring-time-vs-storage-time
issue specific to the write-time-truncated prediction text, not the strict
correctness label itself, which was always scored on full text before the
truncated column was written for storage).

Output: evaluation/item_discrimination_exact.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.novel_methods_utils import load_correctness_matrix  # noqa: E402

N_MODELS = 12
TOTAL_PAIRS = N_MODELS * (N_MODELS - 1) // 2  # 66


def main():
    matrix = load_correctness_matrix()
    assert matrix.shape == (406, 12), f"expected 406x12, got {matrix.shape}"

    k = matrix.sum(axis=1).astype(int)  # items x 1: number of models correct, 0..12
    counts = Counter(int(v) for v in k)
    by_k = {kk: counts.get(kk, 0) for kk in range(13)}

    # pairwise disagreement per item at this k, and the max possible (k=6)
    disagree = {kk: kk * (N_MODELS - kk) for kk in range(13)}

    zero_info = by_k[0] + by_k[12]           # k=0 or k=12: 0 of 66 pairs disagree
    near_ceiling_floor = by_k[11] + by_k[1]  # k=11 or k=1: 11 of 66 pairs disagree
    old_ceiling_213 = by_k[11] + by_k[12]
    old_floor_7 = by_k[0] + by_k[1]

    total_pairwise_disagreement = sum(by_k[kk] * disagree[kk] for kk in range(13))
    max_possible = 406 * TOTAL_PAIRS
    mean_disagreement_frac = total_pairwise_disagreement / max_possible

    result = {
        "n_models": N_MODELS,
        "total_pairs": TOTAL_PAIRS,
        "items_by_k": by_k,
        "pairs_distinguished_by_k": disagree,
        "zero_info_items_k0_or_k12": zero_info,
        "near_ceiling_or_floor_items_k11_or_k1": near_ceiling_floor,
        "old_ceiling_band_k11_or_k12": old_ceiling_213,
        "old_floor_band_k0_or_k1": old_floor_7,
        "total_pairwise_disagreement_sum": int(total_pairwise_disagreement),
        "max_possible_pairwise_disagreement_sum": int(max_possible),
        "mean_disagreement_fraction": round(float(mean_disagreement_frac), 4),
    }
    Path("evaluation/item_discrimination_exact.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print("Items by k (models correct), k=0..12:")
    for kk in range(13):
        print(f"  k={kk:2d}: {by_k[kk]:3d} items, distinguishes {disagree[kk]:2d}/{TOTAL_PAIRS} pairs")
    print(f"\nOld 'ceiling' band (k=11 or 12): {old_ceiling_213} items")
    print(f"  of which truly zero-information (k=12 only): {by_k[12]} items")
    print(f"  of which k=11 (still distinguishes 11/66 pairs each): {by_k[11]} items")
    print(f"\nOld 'floor' band (k=0 or 1): {old_floor_7} items")
    print(f"  of which truly zero-information (k=0 only): {by_k[0]} items")
    print(f"  of which k=1 (still distinguishes 11/66 pairs each): {by_k[1]} items")
    print(f"\nTrue zero-information items (k=0 or 12 only): {zero_info}")
    print(f"Mean pairwise-disagreement fraction across all 406 items: {mean_disagreement_frac:.4f}")
    print("\nSaved -> evaluation/item_discrimination_exact.json")


if __name__ == "__main__":
    main()
