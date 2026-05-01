"""
demo/app.py
-----------
Flask leaderboard + RAG demo for IndiaFinBench.

Routes:
  GET  /                  Main page
  GET  /api/leaderboard   JSON leaderboard (12 models + human)
  GET  /api/example       Random benchmark item
  POST /api/rag           Hybrid RAG query (rate-limited 20/min per IP)
  POST /api/submit        Returns pre-filled GitHub issue URL

Run locally:
  python demo/app.py

On HuggingFace Spaces:
  gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 demo.app:app
"""

import json
import os
import random
import sys
import threading
import time
import urllib.parse
from collections import defaultdict, deque
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# ── Python path ────────────────────────────────────────────────────────────────
# Root must come FIRST so `import rag` resolves to /app/rag/ (the real pipeline)
# not /app/demo/rag/ (old shim, now deleted).
# Demo comes second for database/ imports.
_DEMO_DIR = Path(__file__).parent
_ROOT_DIR  = _DEMO_DIR.parent

if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(1, str(_DEMO_DIR))

from database.db import get_leaderboard, init_db  # noqa: E402

# ── Data ───────────────────────────────────────────────────────────────────────

QUESTIONS_PATH = _DEMO_DIR / "data" / "questions.json"
with QUESTIONS_PATH.open(encoding="utf-8") as _f:
    QUESTIONS: list[dict] = json.load(_f)

init_db()

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(_DEMO_DIR / "templates"),
    static_folder=str(_DEMO_DIR / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

_JS_VER = str(int(time.time()))


@app.after_request
def _no_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


# ── Constants ──────────────────────────────────────────────────────────────────

TASK_FULL = {
    "regulatory_interpretation": "Regulatory Interpretation",
    "numerical_reasoning":       "Numerical Reasoning",
    "contradiction_detection":   "Contradiction Detection",
    "temporal_reasoning":        "Temporal Reasoning",
}

HUMAN_BASELINE = {
    "rank":      "—",
    "label":     "Human Expert",
    "hf_id":     "— (n=100 sampled items)",
    "params":    "—",
    "type":      "Human Baseline",
    "overall":   69.0,
    "reg":       55.6,
    "num":       44.4,
    "con":       83.3,
    "tmp":       66.7,
    "n_items":   100,
    "submitted": "2026-03-15",
    "is_human":  True,
}

# ── RAG pipeline (lazy, thread-safe) ──────────────────────────────────────────

_INDEX_DIR    = _ROOT_DIR / "rag" / "index"
_rag_pipeline = None
_rag_lock     = threading.Lock()


def _get_rag_pipeline():
    """Lazy-init the RAG pipeline with double-checked locking."""
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline
    with _rag_lock:
        if _rag_pipeline is not None:
            return _rag_pipeline
        from rag.config import RAGConfig        # noqa: PLC0415
        from rag.pipeline import RAGPipeline    # noqa: PLC0415
        cfg = RAGConfig(index_dir=_INDEX_DIR)
        p = RAGPipeline(config=cfg)
        p.load_index()
        _rag_pipeline = p
    return _rag_pipeline


def _warmup_rag() -> None:
    """Pre-load the pipeline at startup so the first user request is fast."""
    try:
        _get_rag_pipeline()
        app.logger.info("RAG pipeline ready (FAISS + BM25 loaded)")
    except Exception as exc:                   # noqa: BLE001
        app.logger.warning("RAG warmup failed: %s", exc)


threading.Thread(target=_warmup_rag, daemon=True).start()

# ── Rate limiter ───────────────────────────────────────────────────────────────

_rag_rate: dict = defaultdict(deque)
_RL_N = 20      # requests
_RL_W = 60.0    # seconds window


def _rl_check(ip: str) -> bool:
    """Return True if the IP is within the rate limit, False if exceeded."""
    now = time.time()
    q   = _rag_rate[ip]
    while q and now - q[0] > _RL_W:
        q.popleft()
    if len(q) >= _RL_N:
        return False
    q.append(now)
    return True


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_models(df) -> list[dict]:
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
    return render_template(
        "index.html",
        models=models,
        human=HUMAN_BASELINE,
        model_count=12,
        human_overall=f"{HUMAN_BASELINE['overall']:.1f}",
        js_ver=_JS_VER,
    )


@app.route("/api/leaderboard")
def api_leaderboard():
    df     = get_leaderboard()
    models = _normalize_models(df) if not df.empty else []
    models.append(HUMAN_BASELINE)
    return jsonify(models)


@app.route("/api/submit", methods=["POST"])
def api_submit():
    data       = request.get_json() or {}
    hf_id      = (data.get("hf_id") or "").strip()
    label      = (data.get("label") or (hf_id.split("/")[-1] if hf_id else "")).strip()
    params     = (data.get("params") or "Unknown").strip() or "Unknown"
    model_type = (data.get("model_type") or "Open").strip() or "Open"

    if not hf_id:
        return jsonify({"error": "hf_id is required"}), 400

    safe = hf_id.replace("/", "_")
    body = (
        "**Model Submission for IndiaFinBench**\n\n"
        "| Field | Value |\n|---|---|\n"
        f"| Model Name | {label} |\n"
        f"| HuggingFace ID | `{hf_id}` |\n"
        f"| Parameters | {params} |\n"
        f"| Type | {model_type} |\n\n"
        "**Evaluation command:**\n"
        "```bash\n"
        "python evaluation/evaluate.py \\\n"
        "    --dataset data/benchmark/indiafinbench_v1.csv \\\n"
        f"    --model {hf_id} \\\n"
        "    --provider huggingface \\\n"
        f"    --output results/predictions/{safe}.csv\n"
        "```\n"
    )
    issue_url = (
        "https://github.com/Rajveer-code/IndiaFinBench/issues/new"
        f"?title={urllib.parse.quote(f'[Submission] {label}')}"
        f"&body={urllib.parse.quote(body)}"
        "&labels=model-submission"
    )
    return jsonify({"issue_url": issue_url})


@app.route("/api/rag", methods=["POST"])
def api_rag():
    data  = request.get_json() or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    ip = request.remote_addr or "unknown"
    if not _rl_check(ip):
        return jsonify({"error": "Rate limit reached (20 req/min). Please wait."}), 429

    try:
        result = _get_rag_pipeline().ask(query)
    except Exception as exc:               # noqa: BLE001
        result = {"error": f"RAG unavailable: {str(exc)[:300]}"}
    return jsonify(result)


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
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
