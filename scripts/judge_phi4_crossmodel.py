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

Judge-input truncation (Plan v3 Phase 2.6): this script previously cut its own
inputs to question[:600]/reference[:300]/prediction[:500] on top of whatever
the evaluation harness had already logged (F6). That silent second layer of
truncation is removed. Instead, `_choose_num_ctx()` scans every row about to
be judged, sizes Ollama's context window to the longest observed
question+reference+prediction triple (plus the fixed rubric template and the
num_predict output budget, with a safety margin), and every call asserts the
assembled prompt actually fits before sending it. If a future, longer corpus
(e.g. after Phase 2's untruncated re-run) ever exceeds the assertion, the
row is flagged in the output CSV's `judge_input_truncated` column rather than
silently cut -- this should not fire given `_choose_num_ctx()` sizes from the
same corpus being judged, but a flag beats a silent truncation if it ever does.
The chosen num_ctx and corpus stats are logged to
`evaluation/judge_run_manifest.json`.

Output: evaluation/results_judged_phi4/<model_key>_results.csv with columns
id, task_type, difficulty, question, ref_answer, prediction, strict_correct,
judge_verdict, judge_reason, judge_input_truncated -- kept close to the
existing results_judged/ schema so downstream tooling can compare the two
judges directly.

Resumable: reruns skip ids already present in the output CSV.
"""
import csv
import glob
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.scorer_with_judge_gemini import JUDGE_PROMPT, JUDGE_TASKS  # noqa: E402
from scripts.novel_methods_utils import MODEL_FILES  # noqa: E402

RESULTS_DIR = Path("evaluation/results")
OUT_DIR = Path("evaluation/results_judged_phi4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = Path("evaluation/judge_run_manifest.json")

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "phi4-mini"
NUM_PREDICT = 150
CHARS_PER_TOKEN = 3.0    # conservative (over-)estimate for English legal/financial text
CTX_SAFETY_MARGIN = 1.25  # 25% headroom over the worst-case estimate
CTX_ROUND_TO = 512        # round the computed num_ctx up to a multiple of this

FIELDNAMES = ["id", "task_type", "difficulty", "question", "ref_answer",
              "prediction", "strict_correct", "judge_verdict", "judge_reason",
              "judge_input_truncated"]


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _choose_num_ctx() -> dict:
    """Scan every REG/NUM/TMP row in evaluation/results/*.csv (the full corpus
    this script judges, not just one model's file) and size num_ctx to the
    worst-case assembled prompt, not a value copied from a past run."""
    max_triple_chars = 0
    max_row = None
    n = 0
    for path in glob.glob(str(RESULTS_DIR / "*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("task_type") not in JUDGE_TASKS:
                    continue
                q, r, p = row.get("question") or "", row.get("ref_answer") or "", row.get("prediction") or ""
                total = len(q) + len(r) + len(p)
                n += 1
                if total > max_triple_chars:
                    max_triple_chars = total
                    max_row = (Path(path).name, row.get("id"))
    template_overhead_chars = len(JUDGE_PROMPT.format(question="", reference="", prediction=""))
    worst_case_prompt_tokens = _estimate_tokens(" " * (max_triple_chars + template_overhead_chars))
    needed = math.ceil((worst_case_prompt_tokens + NUM_PREDICT) * CTX_SAFETY_MARGIN)
    num_ctx = max(2048, math.ceil(needed / CTX_ROUND_TO) * CTX_ROUND_TO)
    return {
        "num_ctx": num_ctx,
        "corpus_rows_scanned": n,
        "max_triple_chars": max_triple_chars,
        "max_triple_source": max_row,
        "worst_case_prompt_tokens_est": worst_case_prompt_tokens,
        "num_predict": NUM_PREDICT,
        "chars_per_token_assumed": CHARS_PER_TOKEN,
        "safety_margin": CTX_SAFETY_MARGIN,
    }


NUM_CTX_INFO = _choose_num_ctx()
NUM_CTX = NUM_CTX_INFO["num_ctx"]


def call_phi4_judge(question: str, reference: str, prediction: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)
    prompt_tokens_est = _estimate_tokens(prompt)
    # NUM_CTX is sized from the same corpus this function is called against, so this
    # should never fire; if it ever does (e.g. a row longer than the corpus scan saw),
    # flag it rather than silently truncate.
    input_truncated = (prompt_tokens_est + NUM_PREDICT) > NUM_CTX
    payload = json.dumps({
        "model": JUDGE_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = json.load(r).get("response", "").strip()
            first_line = raw.split("\n")[0].strip().upper()
            verdict = first_line.startswith("CORRECT") and not first_line.startswith("INCORRECT")
            reason = raw.split("Reason:")[-1].strip()[:200] if "Reason:" in raw else raw[:200]
            return {"verdict": verdict, "reason": reason, "input_truncated": input_truncated}
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    retry {attempt + 1}: {e}")
            time.sleep(3)
    return {"verdict": False, "reason": "JUDGE_FAILED", "input_truncated": input_truncated}


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
            "judge_input_truncated": int(v.get("input_truncated", False)),
        })
        if i % 10 == 0 or i == len(todo):
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                w.writeheader()
                w.writerows(out_rows)
            print(f"  [{label}] {i}/{len(todo)} checkpointed")


def main():
    print(f"num_ctx chosen: {NUM_CTX}  ({NUM_CTX_INFO})")
    total_scope = 0
    for key, label, path in model_keys():
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        process_model(key, label, path)
        total_scope += 1

    n_flagged = 0
    for out_path in OUT_DIR.glob("*.csv"):
        with open(out_path, encoding="utf-8") as f:
            n_flagged += sum(1 for r in csv.DictReader(f) if r.get("judge_input_truncated") == "1")

    manifest = {
        **NUM_CTX_INFO,
        "judge_model": JUDGE_MODEL,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "models_processed": total_scope,
        "rows_flagged_judge_input_truncated": n_flagged,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nDone. Processed {total_scope} models. "
          f"{n_flagged} row(s) flagged judge_input_truncated. Manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
