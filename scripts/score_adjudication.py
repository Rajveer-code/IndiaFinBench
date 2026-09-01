"""Score the completed 238-item human-adjudication sheet.

Reports agreement/disagreement evidence only -- deliberately not framed as a
population-level judge error rate (the sample is stratified/oversampled by
design, not a random draw sized for that). See annotation/judge_audit/README.md.

Usage: python scripts/score_adjudication.py [path_to_sheet.csv]
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

SHEET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).parent.parent / "annotation" / "judge_audit" / "adjudication_sheet.csv"

VERDICT_MAP = {"1": "CORRECT", "0": "INCORRECT"}


def main():
    rows = list(csv.DictReader(open(SHEET_PATH, newline="", encoding="utf-8")))
    unfilled = [r for r in rows if not r["human_verdict_CORRECT_or_INCORRECT"].strip()]
    if unfilled:
        print(f"WARNING: {len(unfilled)} rows still unfilled -- scoring the {len(rows)-len(unfilled)} that are done.")
    rows = [r for r in rows if r["human_verdict_CORRECT_or_INCORRECT"].strip()]

    disagreement = [r for r in rows if r["sample_group"] == "disagreement"]
    control = [r for r in rows if r["sample_group"] == "control_agreement"]

    print(f"Total adjudicated: {len(rows)}  (disagreement={len(disagreement)}, control={len(control)})\n")

    # --- Disagreement rows: which judge did the human agree with? ---
    print("=== Disagreement cases (174): which automated judge matched the human verdict ===")
    gemini_match = phi4_match = both_match = neither_match = 0
    by_task = {}
    for r in disagreement:
        h = r["human_verdict_CORRECT_or_INCORRECT"]
        g = VERDICT_MAP.get(r["gemini_verdict"], None)
        p = VERDICT_MAP.get(r["phi4_verdict"], None)
        g_ok = g == h
        p_ok = p == h
        if g_ok and p_ok:
            both_match += 1
        elif g_ok:
            gemini_match += 1
        elif p_ok:
            phi4_match += 1
        else:
            neither_match += 1
        t = r["task_type"]
        by_task.setdefault(t, Counter())[
            "gemini_only" if g_ok and not p_ok else
            "phi4_only" if p_ok and not g_ok else
            "both" if g_ok and p_ok else "neither"
        ] += 1

    n = len(disagreement)
    print(f"  Gemini matched human only: {gemini_match} ({gemini_match/n*100:.1f}%)")
    print(f"  phi4-mini matched human only: {phi4_match} ({phi4_match/n*100:.1f}%)")
    print(f"  Both matched human (human sided with neither's disagreement... i.e. both wrong "
          f"about disagreeing): {both_match} ({both_match/n*100:.1f}%)")
    print(f"  Neither matched human: {neither_match} ({neither_match/n*100:.1f}%)")
    print(f"  By task: {dict((k, dict(v)) for k, v in by_task.items())}")

    # --- Control rows: does human agree with what the methods agreed on? ---
    print("\n=== Control-agreement cases (64): does human agree with the automated consensus ===")
    by_stratum = Counter()
    stratum_total = Counter()
    for r in control:
        h = r["human_verdict_CORRECT_or_INCORRECT"]
        stratum = r["stratum"]
        stratum_total[stratum] += 1
        if "strict_correct" in r and r["strict_correct"] in ("0", "1"):
            consensus = VERDICT_MAP[r["strict_correct"]]
        else:
            consensus = None
        if consensus and h == consensus:
            by_stratum[stratum] += 1

    for stratum in sorted(stratum_total):
        agree = by_stratum[stratum]
        total = stratum_total[stratum]
        print(f"  {stratum}: human agreed with consensus {agree}/{total} ({agree/total*100:.1f}%)")

    total_control_agree = sum(by_stratum.values())
    total_control = sum(stratum_total.values())
    print(f"  TOTAL: human agreed with automated consensus {total_control_agree}/{total_control} "
          f"({total_control_agree/total_control*100:.1f}%)")

    out = {
        "n_disagreement": n, "gemini_matched_only": gemini_match, "phi4_matched_only": phi4_match,
        "both_matched": both_match, "neither_matched": neither_match,
        "disagreement_by_task": {k: dict(v) for k, v in by_task.items()},
        "n_control": total_control, "control_agree_with_consensus": total_control_agree,
        "control_by_stratum": {k: {"agree": by_stratum[k], "total": stratum_total[k]} for k in stratum_total},
    }
    out_path = Path(__file__).parent.parent / "evaluation" / "adjudication_results.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
