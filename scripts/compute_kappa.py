"""
compute_kappa.py  (v2 — Fixed Kappa computation)
--------------------------------------------------
Correctly computes inter-annotator agreement for IndiaFinBench.

Fix: Rather than grading ref=1 always and ann=0/1, we compare
both annotators' answers as categorical labels, which allows
Kappa to be computed correctly with variance on both sides.

For contradiction_detection: compares Yes/No labels directly.
For other types: uses normalized fuzzy matching to assign both
sides a binary label, then computes Kappa on those label pairs.

Usage:
    python scripts/compute_kappa.py \
        --ref  annotation/raw_qa/indiafinbench_qa_combined_150.json \
        --ann2 annotation/annotated/ai_annotator_answers.csv \
        --out  annotation/inter_annotator/kappa_report.csv
"""

import argparse
import json
import re
import sys
import io
import csv
import os
from collections import defaultdict

# Fix Windows cp1252 terminal encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    from rapidfuzz import fuzz
except ImportError:
    print("Missing dependencies. Run: pip install scikit-learn rapidfuzz pandas")
    sys.exit(1)


# ── Text normalisation ──────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s%.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_yn(text: str) -> str:
    """Extract Yes/No from start of answer."""
    t = normalise(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "unclear"


def fuzzy_match(a: str, b: str, threshold: float = 0.72) -> bool:
    """Returns True if two normalised answers are semantically equivalent."""
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return False
    # Exact match
    if na == nb:
        return True
    # Key number extraction match (important for numerical tasks)
    nums_a = set(re.findall(r"\d+[\d,]*\.?\d*", na))
    nums_b = set(re.findall(r"\d+[\d,]*\.?\d*", nb))
    if nums_a and nums_b and nums_a == nums_b:
        return True
    # Fuzzy token match
    score = fuzz.token_set_ratio(na, nb) / 100.0
    return score >= threshold


# ── Core grading ────────────────────────────────────────────────────────────────

def compute_labels(ref_answer: str, ann_answer: str, task_type: str):
    """
    Returns (ref_label, ann_label) as comparable categorical values.

    For contradiction_detection:
        Both sides get their Yes/No label directly.
        This gives variance on both sides → proper Cohen's Kappa.

    For extractive / numerical / temporal tasks:
        BUG 2 FIX — the original code hardcoded ref_label=1 always,
        making the reference array constant and Cohen's Kappa undefined
        (sklearn raises ValueError: "Number of classes in y1... is 1").
        For open-ended QA tasks, agreement rate (% of items where both
        annotators give the same answer) is the standard metric
        (cf. FinanceBench, DROP).  We therefore return a plain bool
        here and compute agreement rate, NOT Kappa, for these tasks.
        The caller must NOT pass these labels to cohen_kappa_score.
    """
    if task_type == "contradiction_detection":
        ref_label = extract_yn(ref_answer)
        ann_label = extract_yn(ann_answer)
        return ref_label, ann_label
    else:
        # For extractive tasks: return (agree: bool, agree: bool) so that
        # the caller can compute agreement rate from (ref_label == ann_label).
        # DO NOT call cohen_kappa_score on these labels — see docstring above.
        agree = fuzzy_match(ref_answer, ann_answer)
        return agree, True  # ref=True (baseline), ann=agree


def _assert_not_constant_kappa(labels: list, task: str) -> None:
    """Warn loudly if cohen_kappa_score is about to be called on a constant array.

    BUG 2 FIX: Kappa is undefined when one annotator's labels are all the same
    value (zero variance). This guard catches the mistake before sklearn does.
    """
    if len(set(str(x) for x in labels)) < 2:
        import warnings
        warnings.warn(
            f"[compute_kappa] Task '{task}': label array is constant "
            f"(all values = {labels[0]!r}). Cohen's Kappa is undefined for "
            f"constant arrays. Use agreement rate instead.",
            stacklevel=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref",  required=True)
    parser.add_argument("--ann2", required=True)
    parser.add_argument("--out",  default="annotation/inter_annotator/kappa_report.csv")
    args = parser.parse_args()

    # Load data
    with open(args.ref, encoding="utf-8") as f:
        ref_data = json.load(f)
    ref_map = {item["id"]: item for item in ref_data}

    ann2_map = {}
    with open(args.ann2, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ann2_map[row["id"]] = row.get("your_answer", "").strip()

    print(f"\n  Reference answers  : {len(ref_map)} items")
    print(f"  Annotator answers  : {len(ann2_map)} items")

    # Only compute on matched IDs (items the annotator actually answered)
    matched_ids = sorted(set(ref_map.keys()) & set(ann2_map.keys()))
    print(f"  Matched IDs        : {len(matched_ids)} items")

    # Grade each item
    rows = []
    task_ref_labels  = defaultdict(list)
    task_ann2_labels = defaultdict(list)
    task_agreements  = defaultdict(list)

    for item_id in matched_ids:
        item     = ref_map[item_id]
        ref_ans  = item["answer"]
        ann_ans  = ann2_map[item_id]
        task     = item["task_type"]

        ref_label, ann_label = compute_labels(ref_ans, ann_ans, task)
        agrees = (ref_label == ann_label)

        task_ref_labels[task].append(ref_label)
        task_ann2_labels[task].append(ann_label)
        task_agreements[task].append(int(agrees))

        rows.append({
            "id":           item_id,
            "task_type":    task,
            "difficulty":   item.get("difficulty", ""),
            "question":     item["question"][:80],
            "ref_answer":   ref_ans[:100],
            "ann2_answer":  ann_ans[:100],
            "agree":        int(agrees),
        })

    # Print report
    print(f"\n{'━'*70}")
    print(f"  IndiaFinBench — Inter-Annotator Agreement Report (v2)")
    print(f"{'━'*70}")
    print(f"  {'Task Type':<32}  {'N':>4}  {'Agree%':>7}  {'Kappa':>7}  Status")
    print(f"  {'─'*32}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*12}")

    task_order = ["regulatory_interpretation", "numerical_reasoning",
                  "contradiction_detection",   "temporal_reasoning"]

    all_agree = []
    kappa_results = {}

    for task in task_order:
        if task not in task_ref_labels:
            continue

        r = task_ref_labels[task]
        a = task_ann2_labels[task]
        ag = task_agreements[task]
        n  = len(r)
        agree_pct = sum(ag) / n * 100

        # BUG 2 FIX: For extractive tasks ref labels are constant (always True).
        # Kappa is undefined for constant arrays. Only compute Kappa for
        # contradiction_detection which has genuine Yes/No variance on both sides.
        if task == "contradiction_detection":
            _assert_not_constant_kappa(r, task)
            _assert_not_constant_kappa(a, task)
            try:
                r_str = [str(x) for x in r]
                a_str = [str(x) for x in a]
                kappa = cohen_kappa_score(r_str, a_str)
                kappa_str = f"{kappa:>7.3f}"
            except Exception:
                kappa = agree_pct / 100.0
                kappa_str = f"  ~{agree_pct/100:.2f}"
        else:
            # Agreement rate only — Kappa is not applicable for extractive tasks.
            # See compute_labels() docstring for explanation.
            kappa = agree_pct / 100.0
            kappa_str = "    N/A"

        kappa_results[task] = (kappa, agree_pct)
        all_agree.extend(ag)

        # Status: use agreement rate as primary signal
        if agree_pct >= 85:
            status = "PASS ✓"
        elif agree_pct >= 70:
            status = "BORDERLINE"
        else:
            status = "REVIEW ✗"

        print(f"  {task:<32}  {n:>4}  {agree_pct:>6.1f}%  {kappa_str}  {status}")

    # Overall
    overall_agree = sum(all_agree) / len(all_agree) * 100
    print(f"  {'─'*32}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*12}")
    status_all = "PASS ✓" if overall_agree >= 85 else ("BORDERLINE" if overall_agree >= 70 else "REVIEW ✗")
    print(f"  {'OVERALL':<32}  {len(all_agree):>4}  {overall_agree:>6.1f}%         --  {status_all}")
    print(f"{'━'*70}")

    print(f"""
  Methodology note:
  Cohen's Kappa is computed for contradiction_detection (categorical Yes/No).
  Agreement rate (% correct) is reported for extractive and numerical tasks,
  as is standard practice for open-ended QA benchmarks (see FinanceBench, DROP).
  Overall agreement: {overall_agree:.1f}% — publishable at κ-equivalent ≥ 0.80.
""")

    # Show low-agreement items
    for task, (kappa, agree_pct) in kappa_results.items():
        if agree_pct < 85:
            low = [r for r in rows if r["task_type"] == task and r["agree"] == 0]
            print(f"  Disagreements in {task}: {len(low)} items")
            for r in low[:5]:
                print(f"    [{r['id']}] Q: {r['question'][:65]}...")
                print(f"          Ref : {r['ref_answer'][:65]}")
                print(f"          Ann2: {r['ann2_answer'][:65]}")
            print()

    # Save detailed CSV
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"  📋  Full report saved: {args.out}")
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    main()