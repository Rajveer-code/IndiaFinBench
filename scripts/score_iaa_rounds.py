"""
score_iaa_rounds.py
--------------------
Plan v3 Phase 6.1: score IAA rounds 2 and 3 with the same four-stage strict
scorer used for models (scripts/evaluate.py::score_answer -- exact match via
normalisation, numeric-set match, RapidFuzz token_set_ratio >= 0.72, Yes/No
match for contradiction_detection). Round 1 was already scored to disk
(annotation/iaa/iaa_60item_sample.csv has a pre-computed `agree` column);
rounds 2 and 3 were collected but never scored.

Round 1: annotation/iaa/iaa_60item_sample.csv (id, ref_answer, ann2_answer, agree)
Round 2: annotation/iaa/iaa_expansion_annotator2.csv (item_id, YOUR_ANSWER)
         + iaa_expansion_reference.csv (item_id, reference_answer)
Round 3: annotation/iaa/iaa_60item_FILLED_FINAL.csv (item_id, YOUR_ANSWER),
         reference looked up from the main QA dataset by item_id.

All 180 ids are disjoint across rounds (verified below) and all resolve in
annotation/raw_qa/indiafinbench_qa_combined_406.json.

Output: annotation/iaa/iaa_180item_scored.csv (round, item_id, task_type,
ref_answer, ann2_answer, agree) + evaluation/iaa_summary.json (overall and
per-task agreement, plus Cohen's kappa on the binary CON task, pooled and
per round).
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate import score_answer, normalise  # noqa: E402 -- canonical four-stage scorer
from sklearn.metrics import cohen_kappa_score  # noqa: E402 -- matches scripts/compute_kappa.py's methodology

BASE = Path(__file__).parent.parent
QA_PATH = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
IAA_DIR = BASE / "annotation/iaa"
OUT_CSV = IAA_DIR / "iaa_180item_scored.csv"
OUT_JSON = BASE / "evaluation/iaa_summary.json"

TASK_MAP = {
    "regulatory_interpretation": "REG", "REG": "REG",
    "numerical_reasoning": "NUM", "NUM": "NUM",
    "contradiction_detection": "CON", "CON": "CON",
    "temporal_reasoning": "TMP", "TMP": "TMP",
}


def load_qa():
    with open(QA_PATH, encoding="utf-8") as f:
        return {item["id"]: item for item in json.load(f)}


def score_round1(qa):
    rows = []
    with open(IAA_DIR / "iaa_60item_sample.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "round": 1, "item_id": r["id"], "task_type": TASK_MAP[r["task_type"]],
                "ref_answer": r["ref_answer"], "ann2_answer": r["ann2_answer"],
                "agree": int(r["agree"]),
            })
    return rows


def score_round2():
    ref = {}
    with open(IAA_DIR / "iaa_expansion_reference.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[r["item_id"]] = r["reference_answer"]
    rows = []
    with open(IAA_DIR / "iaa_expansion_annotator2.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = r["item_id"]
            ref_ans = ref.get(iid, "")
            ann_ans = r["YOUR_ANSWER"]
            task = TASK_MAP[r["task_type"]]
            agree = score_answer(ref_ans, ann_ans, r["task_type"])
            rows.append({
                "round": 2, "item_id": iid, "task_type": task,
                "ref_answer": ref_ans, "ann2_answer": ann_ans, "agree": agree,
            })
    return rows


def score_round3(qa):
    rows = []
    with open(IAA_DIR / "iaa_60item_FILLED_FINAL.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = r["item_id"]
            item = qa.get(iid)
            if item is None:
                raise ValueError(f"round 3 item_id {iid} not found in QA dataset")
            ref_ans = item["answer"]
            ann_ans = r["YOUR_ANSWER"]
            task = TASK_MAP[r["task_type"]]
            agree = score_answer(ref_ans, ann_ans, r["task_type"])
            rows.append({
                "round": 3, "item_id": iid, "task_type": task,
                "ref_answer": ref_ans, "ann2_answer": ann_ans, "agree": agree,
            })
    return rows


def extract_yn(text: str) -> str:
    """Yes/No extraction for CON kappa -- matches scripts/compute_kappa.py's
    extract_yn exactly (the method that produced the round-1 kappa=0.645
    figure currently in the manuscript)."""
    t = normalise(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "unclear"


def con_kappa(rows):
    """Cohen's kappa on CON items only, computed from independently-extracted
    Yes/No labels on both the reference and the annotator's answer (both sides
    have real variance, unlike scoring 'agree' against a constant reference) --
    matches scripts/compute_kappa.py's methodology."""
    ref_labels, ann_labels = [], []
    for r in rows:
        if r["task_type"] != "CON":
            continue
        ry, ay = extract_yn(r["ref_answer"]), extract_yn(r["ann2_answer"])
        if ry == "unclear" or ay == "unclear":
            continue
        ref_labels.append(ry)
        ann_labels.append(ay)
    if len(ref_labels) < 2 or len(set(ref_labels)) < 2 or len(set(ann_labels)) < 2:
        return None, len(ref_labels)
    return cohen_kappa_score(ref_labels, ann_labels), len(ref_labels)


def main():
    qa = load_qa()
    r1, r2, r3 = score_round1(qa), score_round2(), score_round3(qa)
    all_rows = r1 + r2 + r3

    ids = [r["item_id"] for r in all_rows]
    assert len(ids) == len(set(ids)), "duplicate item_id across IAA rounds"
    for r in all_rows:
        assert r["item_id"] in qa, f"{r['item_id']} not in main QA dataset"
    assert len(all_rows) == 180, f"expected 180 total IAA items, got {len(all_rows)}"

    IAA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["round", "item_id", "task_type", "ref_answer", "ann2_answer", "agree"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows -> {OUT_CSV}")

    def summarize(rows, label):
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_type"]].append(r["agree"])
        overall = sum(r["agree"] for r in rows) / len(rows) if rows else float("nan")
        per_task = {t: sum(v) / len(v) for t, v in by_task.items()}
        kappa, kappa_n = con_kappa(rows)
        out = {
            "n": len(rows), "overall_agreement": overall, "per_task_agreement": per_task,
            "con_kappa": kappa, "con_kappa_n": kappa_n,
        }
        print(f"\n{label}: n={out['n']}  overall={out['overall_agreement']:.3f}"
              f"  CON kappa={kappa if kappa is None else f'{kappa:.3f}'} (n={kappa_n})")
        for t, v in sorted(out["per_task_agreement"].items()):
            print(f"  {t}: {v:.3f} (n={len(by_task[t])})")
        return out

    summary = {
        "round1": summarize(r1, "Round 1 (previously scored, frozen)"),
        "round2": summarize(r2, "Round 2 (newly scored, four-stage scorer)"),
        "round3": summarize(r3, "Round 3 (newly scored, four-stage scorer)"),
        "pooled_180": summarize(all_rows, "Pooled (all 180)"),
        "methodology_note": (
            "Round 1's `agree` column (annotation/iaa/iaa_60item_sample.csv) is frozen prior "
            "work that does not exactly reproduce under scripts/evaluate.py::score_answer alone "
            "(46/60=76.7% under the plain four-stage scorer vs. the published 85.0%): round 1 "
            "additionally applied scripts/fix_num_iaa.py, a NUM-specific regex normalizer, on top "
            "of the base scorer. Rounds 2 and 3 (this script) are scored with the plain four-stage "
            "scorer only, with no post-hoc normalization pass. The pooled 180-item figure below "
            "therefore combines two passes that are close but not identical in method; this is "
            "disclosed rather than papered over, consistent with how this paper treats scoring-rule "
            "sensitivity everywhere else."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary -> {OUT_JSON}")


if __name__ == "__main__":
    main()
