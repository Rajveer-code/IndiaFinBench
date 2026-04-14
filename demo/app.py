"""
demo/app.py
-----------
Purpose:  Flask-based IndiaFinBench leaderboard with polished, custom HTML/CSS/JS UI.
          GET  /              — Main leaderboard page
          GET  /api/leaderboard  — JSON leaderboard data
          POST /api/submit    — Kick off async HF model evaluation
          GET  /api/job/<id>  — Poll job status
          GET  /api/example   — Random dataset example (for dataset explorer)
Inputs:   demo/data/questions.json, demo/data/baselines.json, demo/leaderboard.db
Outputs:  Flask web app served on 0.0.0.0:7860
Usage:
    python demo/app.py
"""

import json
import random
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Ensure demo/ is on the Python path when running from repo root
_DEMO_DIR = Path(__file__).parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from database.db import get_leaderboard, init_db, save_result
from evaluation.scorer import score_submission

# ── Globals ────────────────────────────────────────────────────────────────────

QUESTIONS_PATH = _DEMO_DIR / "data/questions.json"
with QUESTIONS_PATH.open(encoding="utf-8") as _f:
    QUESTIONS: list[dict] = json.load(_f)

init_db()

app = Flask(__name__, template_folder=str(_DEMO_DIR / "templates"))

_eval_lock = threading.Lock()
_eval_jobs: dict[str, dict] = {}   # job_id → status dict

TASK_FULL = {
    "regulatory_interpretation": "Regulatory Interpretation",
    "numerical_reasoning":       "Numerical Reasoning",
    "contradiction_detection":   "Contradiction Detection",
    "temporal_reasoning":        "Temporal Reasoning",
}

HUMAN_BASELINE = {
    "rank":      "—",
    "label":     "Human Expert",
    "hf_id":     "— (n=30 sampled items)",
    "params":    "—",
    "type":      "Human Baseline",
    "overall":   60.0,
    "reg":       55.6,
    "num":       44.4,
    "con":       83.3,
    "tmp":       66.7,
    "n_items":   30,
    "submitted": "2026-03-15",
    "is_human":  True,
}


def _normalize_models(df) -> list[dict]:
    """Convert get_leaderboard() DataFrame to JSON-serialisable list."""
    result = []
    for _, row in df.iterrows():
        result.append({
            "rank":      int(row["Rank"]),
            "label":     str(row["Model"]),
            "hf_id":     str(row["HF Model ID"]),
            "params":    str(row.get("Params", "—")),
            "type":      str(row.get("Type", "Open")),
            "overall":   round(float(row["Overall (%)"]), 1),
            "reg":       round(float(row["REG (%)"]),     1),
            "num":       round(float(row["NUM (%)"]),     1),
            "con":       round(float(row["CON (%)"]),     1),
            "tmp":       round(float(row["TMP (%)"]),     1),
            "submitted": str(row.get("Submitted", "")),
            "is_human":  False,
        })
    return result


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    df     = get_leaderboard()
    models = _normalize_models(df) if not df.empty else []
    return render_template("index.html", models=models, human=HUMAN_BASELINE)


@app.route("/api/leaderboard")
def api_leaderboard():
    df     = get_leaderboard()
    models = _normalize_models(df) if not df.empty else []
    models.append(HUMAN_BASELINE)
    return jsonify(models)


@app.route("/api/submit", methods=["POST"])
def api_submit():
    data  = request.get_json() or {}
    hf_id = data.get("hf_id", "").strip()
    label = data.get("label", "").strip() or (hf_id.split("/")[-1] if hf_id else "")
    params     = data.get("params", "").strip() or "Unknown"
    model_type = data.get("model_type", "Open").strip() or "Open"
    smoke      = bool(data.get("smoke", False))

    if not hf_id:
        return jsonify({"error": "Missing hf_id"}), 400

    job_id = str(uuid.uuid4())
    _eval_jobs[job_id] = {
        "status":   "queued",
        "progress": 0,
        "result":   None,
        "error":    None,
    }

    def _run():
        if not _eval_lock.acquire(blocking=False):
            _eval_jobs[job_id].update(
                status="error",
                error="Another evaluation is running. Please try again shortly.",
            )
            return
        try:
            _eval_jobs[job_id]["status"] = "running"
            from evaluation.evaluator import IndiaFinBenchEvaluator

            items     = QUESTIONS[:10] if smoke else QUESTIONS
            evaluator = IndiaFinBenchEvaluator(hf_id)
            preds     = evaluator.run(items)
            result    = score_submission(preds, items)

            save_result(
                hf_id=hf_id,
                label=label,
                overall=result["overall"],
                per_task=result["per_task"],
                params=params,
                model_type=model_type,
                n_items=len(items),
                notes="smoke_test" if smoke else "",
            )
            _eval_jobs[job_id].update(
                status="done",
                result={
                    "label":    label,
                    "overall":  round(result["overall"] * 100, 1),
                    "per_task": {k: round(v * 100, 1) for k, v in result["per_task"].items()},
                    "n_items":  len(items),
                },
            )
        except Exception as exc:                          # noqa: BLE001
            _eval_jobs[job_id].update(status="error", error=str(exc)[:400])
        finally:
            _eval_lock.release()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    job = _eval_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/example")
def api_example():
    task = request.args.get("task", "All")
    diff = request.args.get("diff", "All")

    pool = list(QUESTIONS)
    if task != "All":
        pool = [q for q in pool if TASK_FULL.get(q["task_type"], "") == task]
    if diff != "All":
        pool = [q for q in pool if q["difficulty"] == diff.lower()]

    if not pool:
        return jsonify({"error": "No examples match filters"})

    q   = random.choice(pool)
    ctx = q.get("context") or (
        "Passage A: " + q.get("context_a", "")
        + "\n\nPassage B: " + q.get("context_b", "")
    )
    return jsonify({
        "id":         q["id"],
        "task_type":  TASK_FULL.get(q["task_type"], q["task_type"]),
        "difficulty": q["difficulty"],
        "context":    ctx[:800] + ("…" if len(ctx) > 800 else ""),
        "question":   q["question"],
        "answer":     q["gold_answer"],
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
