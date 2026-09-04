"""
score_independent_adjudication.py
------------------------------
Plan v3 Phase 7: two independent, blind human adjudicators (recruited by the
author, not involved in building the benchmark) completed the full 62-item
stratified blind sample -- annotation/independent_adjudication/blind_sheet.csv
-- with no access to model identity, judge verdicts, or each other's answers.
Referred to only as Adjudicator A / Adjudicator B throughout -- per the paper's
Ethics Statement, no identifying participant information is released, so no
real name appears in this script, its file paths, its output, or the paper.

Computes, from real completed data (Guardrail 10 -- traceable, re-runnable):
- Raw agreement and Cohen's kappa between the two independent adjudicators.
- Each adjudicator's raw agreement with the author's own blind verdict
  (sample_key.csv::author_verdict, from the frozen 238-item adjudication
  sheet -- never modified, only read).
- Each adjudicator's raw agreement with each judge (Gemini, phi4-mini).
- A majority-of-three (author + two independent adjudicators) reading against
  each judge, since three independent verdicts is a materially stronger
  reference point than one.
- Per-task (REG/NUM/TMP) breakdown of all of the above.

All four source files (blind_sheet.csv, sample_key.csv, and the two completed
CSVs) are matched by ROW POSITION, not item_id -- the sample deliberately
contains repeated item_ids (same question, different models' disputed
predictions), so item_id alone is not a unique key. Confirmed all four files
share identical row order before this script was written.

Usage: python scripts/score_independent_adjudication.py
Output: evaluation/independent_adjudication_results.json
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
SAMPLE_KEY = BASE / "annotation/independent_adjudication/sample_key.csv"
ADJUDICATOR_A = BASE / "annotation/independent_adjudication/completed/adjudicator_a.csv"
ADJUDICATOR_B = BASE / "annotation/independent_adjudication/completed/adjudicator_b.csv"
OUT = BASE / "evaluation/independent_adjudication_results.json"


def load(path, col):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return [r[col].strip().upper() for r in rows]


def cohens_kappa(a, b):
    """Binary Cohen's kappa over two equal-length CORRECT/INCORRECT lists."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    a_correct = sum(1 for x in a if x == "CORRECT") / n
    b_correct = sum(1 for x in b if x == "CORRECT") / n
    pe = a_correct * b_correct + (1 - a_correct) * (1 - b_correct)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def agree_rate(a, b):
    return round(100 * sum(1 for x, y in zip(a, b) if x == y) / len(a), 1)


def main():
    key_rows = list(csv.DictReader(open(SAMPLE_KEY, encoding="utf-8-sig")))
    tasks = [r["task"] for r in key_rows]
    author = [r["author_verdict"].strip().upper() for r in key_rows]
    gemini = ["CORRECT" if r["gemini_verdict"] == "1" else "INCORRECT" for r in key_rows]
    phi4 = ["CORRECT" if r["phi4_verdict"] == "1" else "INCORRECT" for r in key_rows]
    adj_a = load(ADJUDICATOR_A, "your_verdict_CORRECT_or_INCORRECT")
    adj_b = load(ADJUDICATOR_B, "your_verdict_CORRECT_or_INCORRECT")

    n = len(key_rows)
    assert n == 62 and len(adj_a) == n and len(adj_b) == n

    majority3 = []
    for a, x, y in zip(author, adj_a, adj_b):
        votes = [a, x, y]
        majority3.append("CORRECT" if votes.count("CORRECT") >= 2 else "INCORRECT")

    result = {
        "n": n,
        "pairwise": {
            "adjudicator_a_vs_adjudicator_b": {"agree_pct": agree_rate(adj_a, adj_b), "kappa": round(cohens_kappa(adj_a, adj_b), 3)},
            "adjudicator_a_vs_author": {"agree_pct": agree_rate(adj_a, author), "kappa": round(cohens_kappa(adj_a, author), 3)},
            "adjudicator_b_vs_author": {"agree_pct": agree_rate(adj_b, author), "kappa": round(cohens_kappa(adj_b, author), 3)},
        },
        "vs_judges": {
            "adjudicator_a_vs_gemini": agree_rate(adj_a, gemini),
            "adjudicator_a_vs_phi4": agree_rate(adj_a, phi4),
            "adjudicator_b_vs_gemini": agree_rate(adj_b, gemini),
            "adjudicator_b_vs_phi4": agree_rate(adj_b, phi4),
            "author_vs_gemini": agree_rate(author, gemini),
            "author_vs_phi4": agree_rate(author, phi4),
            "majority3_vs_gemini": agree_rate(majority3, gemini),
            "majority3_vs_phi4": agree_rate(majority3, phi4),
        },
        "per_task": {},
    }

    for task_code, task_name in [("REG", "regulatory_interpretation"), ("NUM", "numerical_reasoning"), ("TEM", "temporal_reasoning")]:
        idx = [i for i, t in enumerate(tasks) if t == task_code]
        if not idx:
            continue
        sub_a = [adj_a[i] for i in idx]
        sub_b = [adj_b[i] for i in idx]
        sub_author = [author[i] for i in idx]
        result["per_task"][task_name] = {
            "n": len(idx),
            "adjudicator_a_vs_adjudicator_b": agree_rate(sub_a, sub_b),
            "adjudicator_a_vs_author": agree_rate(sub_a, sub_author),
            "adjudicator_b_vs_author": agree_rate(sub_b, sub_author),
        }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"n = {n}")
    print("\n=== Pairwise agreement ===")
    for k, v in result["pairwise"].items():
        print(f"  {k}: {v['agree_pct']}% agreement, kappa={v['kappa']}")
    print("\n=== Agreement with judges ===")
    for k, v in result["vs_judges"].items():
        print(f"  {k}: {v}%")
    print("\n=== Per-task (A/B/Author pairwise) ===")
    for task, v in result["per_task"].items():
        print(f"  {task} (n={v['n']}): A-B={v['adjudicator_a_vs_adjudicator_b']}%  "
              f"A-Author={v['adjudicator_a_vs_author']}%  B-Author={v['adjudicator_b_vs_author']}%")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
