"""
build_independent_adjudication_sample.py
------------------------------------------
Plan v3 Phase 7.1-7.2 / cleanup item 2 (prep half only -- this builds the
blind stratified sample; the actual adjudication needs a real person who did
not build the benchmark, which this script cannot and does not attempt).

Draws a stratified subset of the 174 Gemini/phi4-mini disagreement items
already present in annotation/judge_audit/adjudication_sheet.csv (frozen,
Guardrail 2 -- read-only here, never rewritten), targeting >=60 items,
explicitly stratified by:
  - task (REG / NUM / TMP)
  - disagreement direction (which judge said correct)
  - strict-correct vs strict-incorrect status
  - short vs long prediction (median split within this pool)
  - truncated vs untruncated prediction

Writes a genuinely blind sheet: no model identity, no judge verdicts, no
strict result, no author's own prior verdict on these same items --
question / reference / prediction only. The stratification key is kept in a
SEPARATE file so it can be joined back after adjudication without ever being
visible to the adjudicator during review.

Outputs:
  annotation/independent_adjudication/blind_sheet.csv   (item_id, question,
    reference_answer, model_prediction -- hand this to the adjudicator)
  annotation/independent_adjudication/sample_key.csv     (item_id + every
    stratification field + all three original verdicts -- keep this closed
    until adjudication is complete)
"""
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path("annotation/judge_audit/adjudication_sheet.csv")
OUT_DIR = Path("annotation/independent_adjudication")
BLIND_OUT = OUT_DIR / "blind_sheet.csv"
KEY_OUT = OUT_DIR / "sample_key.csv"
TARGET_N = 62
SEED = 7  # distinct from the 238-sheet's own construction seed


def load_disagreement_rows():
    with open(SRC, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["stratum"].endswith("_disagreement")]
    assert len(rows) == 174, f"expected 174 disagreement rows, found {len(rows)}"
    return rows


def stratify_key(row: dict, median_len: float) -> tuple:
    task = row["task_type"][:3].upper()
    direction = "gemini_correct" if row["gemini_verdict"] == "1" else "phi4_correct"
    strict = "strict_correct" if row["strict_correct"] == "1" else "strict_incorrect"
    length_bucket = "long" if len(row["model_prediction"]) >= median_len else "short"
    truncated = "truncated" if row["prediction_truncated"].upper() == "TRUE" else "untruncated"
    return (task, direction, strict, length_bucket, truncated)


def main():
    rows = load_disagreement_rows()
    median_len = statistics.median(len(r["model_prediction"]) for r in rows)

    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[stratify_key(r, median_len)].append(r)

    print(f"{len(rows)} disagreement items, {len(by_stratum)} distinct strata cells, "
          f"median prediction length {median_len:.0f} chars")
    for k, v in sorted(by_stratum.items()):
        print(f"  {k}: {len(v)}")

    rng = random.Random(SEED)
    # Proportional allocation per cell, rounded up, then trimmed/topped-up to hit TARGET_N
    # exactly -- guarantees every non-empty cell contributes at least one item rather than
    # letting a small cell round to zero and vanish from the sample.
    sample = []
    remaining_by_cell = {k: list(v) for k, v in by_stratum.items()}
    for k in remaining_by_cell:
        rng.shuffle(remaining_by_cell[k])
    quota = {k: max(1, round(len(v) / len(rows) * TARGET_N)) for k, v in by_stratum.items()}
    for k, q in quota.items():
        take = remaining_by_cell[k][:q]
        sample.extend(take)
        remaining_by_cell[k] = remaining_by_cell[k][q:]

    # Trim/top-up to land exactly on TARGET_N, drawing extras from whatever cells still
    # have items left (largest cells first) rather than from a fixed cell every time.
    all_leftover = [r for cell in remaining_by_cell.values() for r in cell]
    rng.shuffle(all_leftover)
    while len(sample) < TARGET_N and all_leftover:
        sample.append(all_leftover.pop())
    while len(sample) > TARGET_N:
        sample.pop(rng.randrange(len(sample)))

    rng.shuffle(sample)  # presentation order is not stratum-grouped

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BLIND_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "task_type", "question", "reference_answer",
                                           "model_prediction", "your_verdict_CORRECT_or_INCORRECT", "notes"])
        w.writeheader()
        for r in sample:
            w.writerow({
                "item_id": r["item_id"], "task_type": r["task_type"], "question": r["question"],
                "reference_answer": r["reference_answer"], "model_prediction": r["model_prediction"],
                "your_verdict_CORRECT_or_INCORRECT": "", "notes": "",
            })

    with open(KEY_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["item_id", "task", "direction", "strict_status", "length_bucket",
                      "truncated", "model", "strict_correct", "gemini_verdict", "phi4_verdict",
                      "author_verdict"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sample:
            k = stratify_key(r, median_len)
            w.writerow({
                "item_id": r["item_id"], "task": k[0], "direction": k[1], "strict_status": k[2],
                "length_bucket": k[3], "truncated": k[4], "model": r["model"],
                "strict_correct": r["strict_correct"], "gemini_verdict": r["gemini_verdict"],
                "phi4_verdict": r["phi4_verdict"],
                "author_verdict": r["human_verdict_CORRECT_or_INCORRECT"],
            })

    print(f"\nWrote {len(sample)} items -> {BLIND_OUT} (hand this to the adjudicator, nothing else)")
    print(f"Wrote stratification key -> {KEY_OUT} (keep closed until adjudication is complete)")

    final_by_cell = defaultdict(int)
    for r in sample:
        final_by_cell[stratify_key(r, median_len)] += 1
    print("\nFinal sample stratification:")
    for k, n in sorted(final_by_cell.items()):
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
