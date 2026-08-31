"""Full-coverage cross-model judge audit using a locally-run phi4-mini.

Why this exists (Phase 2 of the TMLR resubmission plan): the paper's existing
judge is Gemini 2.5 Flash, which is simultaneously one of the 12 *evaluated*
models. That is a real objection -- a judge should not also be a subject.
phi4-mini (Microsoft Phi) is the one model family with zero overlap with any
of the 12 benchmarked models, runs locally on Ollama, and costs nothing.

Differs from scripts/scorer_with_judge_gemini.py in two ways, both
deliberate:
  1. Full coverage -- judges every REG/NUM/TMP item for every model, not only
     the items strict scoring already marked wrong. This is the only way to
     estimate a strict *false-positive* rate, which the original audit design
     could not produce (it only ever looked at strict failures).
  2. CON is still excluded, matching the paper's existing rationale: CON uses
     exact Yes/No matching against an unambiguous binary label and does not
     need semantic review. This is a pre-existing, principled design choice
     in the codebase, not a new gap.

Reuses JUDGE_PROMPT verbatim from scorer_with_judge_gemini.py so the rubric
authors already validated is what phi4-mini is applying too -- the swap is
the judge, not the standard.

Output: evaluation/results_judged_phi4/<model_key>_results.csv with columns
id, task_type, difficulty, question, ref_answer, prediction, strict_correct,
judge_verdict, judge_reason -- kept close to the existing results_judged/
schema so downstream tooling can compare the two judges directly.

Resumable: reruns skip ids already present in the output CSV.
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.scorer_with_judge_gemini import JUDGE_PROMPT, JUDGE_TASKS  # noqa: E402
from scripts.novel_methods_utils import MODEL_FILES  # noqa: E402

RESULTS_DIR = Path("evaluation/results")
OUT_DIR = Path("evaluation/results_judged_phi4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "phi4-mini"

FIELDNAMES = ["id", "task_type", "difficulty", "question", "ref_answer",
              "prediction", "strict_correct", "judge_verdict", "judge_reason"]


def call_phi4_judge(question: str, reference: str, prediction: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question[:600], reference=reference[:300], prediction=prediction[:500],
    )
    payload = json.dumps({
        "model": JUDGE_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": 150},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = json.load(r).get("response", "").strip()
            first_line = raw.split("\n")[0].strip().upper()
            verdict = first_line.startswith("CORRECT") and not first_line.startswith("INCORRECT")
            reason = raw.split("Reason:")[-1].strip()[:200] if "Reason:" in raw else raw[:200]
            return {"verdict": verdict, "reason": reason}
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    retry {attempt + 1}: {e}")
            time.sleep(3)
    return {"verdict": False, "reason": "JUDGE_FAILED"}


def model_keys():
    # MODEL_FILES: "Display Name" -> "file_results.csv"; derive the short key
    # from the filename so output files match the existing repo convention.
    for label, fname in MODEL_FILES.items():
        key = fname.replace("_results.csv", "")
        yield key, label, RESULTS_DIR / fname


def process_model(key: str, label: str, csv_path: Path):
    out_path = OUT_DIR / f"{key}_results.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done_ids = {r["id"] for r in csv.DictReader(f)}

    todo = [r for r in rows if r["id"] not in done_ids and r["task_type"] in JUDGE_TASKS]
    print(f"[{label}] {len(rows)} rows, {len(todo)} to judge "
          f"({len(done_ids)} already done, {len(rows) - len([r for r in rows if r['task_type'] in JUDGE_TASKS])} CON excluded)")

    out_rows = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            out_rows = list(csv.DictReader(f))

    for i, row in enumerate(todo, 1):
        v = call_phi4_judge(row["question"], row["ref_answer"], row["prediction"])
        out_rows.append({
            "id": row["id"], "task_type": row["task_type"], "difficulty": row["difficulty"],
            "question": row["question"], "ref_answer": row["ref_answer"],
            "prediction": row["prediction"], "strict_correct": row["correct"],
            "judge_verdict": int(v["verdict"]), "judge_reason": v["reason"],
        })
        if i % 10 == 0 or i == len(todo):
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                w.writeheader()
                w.writerows(out_rows)
            print(f"  [{label}] {i}/{len(todo)} checkpointed")


def main():
    total_scope = 0
    for key, label, path in model_keys():
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        process_model(key, label, path)
        total_scope += 1
    print(f"\nDone. Processed {total_scope} models.")


if __name__ == "__main__":
    main()
