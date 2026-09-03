"""
matched_budget_rerun.py
-------------------------
Plan v3 Phase 2.2 / cleanup item 1: the full matched-budget re-run, at the
budget selected from scripts/matched_budget_pilot.py's pilot data.

Budget decision (2026-09-03): 512. Pilot mean truncation across 10 fully-piloted
models was 1.1% at budget=512, 0.4% at 1024, 0.0% at 2048 -- raising the budget
past 512 buys almost nothing (a ~1pp reduction in an already-negligible rate),
so 512 is the lowest budget that does not materially increase truncation
relative to higher budgets. See evaluation/matched_budget_pilot.json for the
full per-model table this was read from.

Reuses the exact caller functions from matched_budget_pilot.py (Groq/OpenRouter/
Ollama/Gemini, including the truststore SSL fix and the Qwen3-32B /no_think
suffix) -- does not reimplement them. Extends build_prompt with the CON
(contradiction_detection) branch, which the pilot's stratified REG/NUM/TMP-only
sample didn't need.

This is a SENSITIVE ANALYSIS, not a replacement: writes to a NEW directory
(evaluation/results_matched/), never touches evaluation/results/ (the original,
frozen release this whole paper's other numbers trace to).

Logs finish_reason and completion_tokens per row (the permanent fix for F6 --
the original release has neither).

Usage:
  python scripts/matched_budget_rerun.py                    # all reachable models
  python scripts/matched_budget_rerun.py --models "LLaMA-3-8B" "Mistral-7B"
"""
import argparse, csv, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from matched_budget_pilot import (  # noqa: E402 -- reuse, don't reimplement
    MODELS, EXCLUDED, CALLERS, SYSTEM_PROMPT, GEMINI_API_KEYS,
)

BASE = Path(__file__).parent.parent
QA_PATH = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
OUT_DIR = BASE / "evaluation/results_matched"
BUDGET = 512
FIELDS = ["id", "task_type", "difficulty", "question", "ref_answer", "prediction",
          "finish_reason", "completion_tokens", "error", "model_version"]

DELAY_MAP = {"ollama": 0.3, "groq": 1.5, "openrouter": 2.0, "gemini": 15.0}
# Gemini: pilot run hit sustained 429s ("limit: 20, generate_content_free_tier_requests")
# even at 8s spacing -- this fresh key's effective free-tier rate is well under the
# documented 15-20 RPM for gemini-2.5-flash, consistent with new-project throttling.
# 15s spacing (~4 req/min) plus call_gemini's own retry-after parsing is the compromise
# between finishing this century and not spending the whole run in backoff.


def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q = item["question"]
    if task == "contradiction_detection":
        return (
            f"Passage A:\n{item.get('context_a', '')[:1500]}\n\n"
            f"Passage B:\n{item.get('context_b', '')[:1500]}\n\n"
            f"Question: {q}\n\nAnswer with 'Yes' or 'No' then one sentence of explanation:"
        )
    ctx = " ".join(item.get("context", "").split()[:450])
    if task == "numerical_reasoning":
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nShow your calculation and give the final answer with units:"
    elif task == "temporal_reasoning":
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer precisely, noting relevant dates or sequences:"
    else:
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def model_key(label: str) -> str:
    return label.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def run_model(label: str, cfg: dict, data: list):
    out_path = OUT_DIR / f"{model_key(label)}_results.csv"
    caller = CALLERS[cfg["provider"]]

    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("error"):
                    done[row["id"]] = row

    todo = [item for item in data if item["id"] not in done]
    print(f"\n[{label}] provider={cfg['provider']} model_id={cfg['model_id']} "
          f"budget={BUDGET} -- {len(done)}/{len(data)} done, {len(todo)} to go")

    rows = list(done.values())
    for i, item in enumerate(todo, 1):
        prompt = build_prompt(item) + cfg.get("prompt_suffix", "")
        out = caller(cfg["model_id"], prompt, BUDGET)
        rows.append({
            "id": item["id"], "task_type": item["task_type"], "difficulty": item.get("difficulty", ""),
            "question": item["question"], "ref_answer": item["answer"],
            "prediction": out.get("text", ""), "finish_reason": out.get("finish_reason"),
            "completion_tokens": out.get("completion_tokens"), "error": out.get("error"),
            "model_version": cfg["model_id"],
        })
        status = "ERR" if out.get("error") else "ok"
        print(f"  [{i}/{len(todo)}] {item['id']:<10} {status}")
        if i % 20 == 0 or i == len(todo):
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
        time.sleep(DELAY_MAP.get(cfg["provider"], 2.0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"[{label}] done: {len(rows)} rows, {n_err} errors -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items. Budget={BUDGET}. Output -> {OUT_DIR}")
    print(f"Excluded: {EXCLUDED}")

    targets = args.models or list(MODELS.keys())
    for label in targets:
        if label not in MODELS:
            print(f"SKIP unknown model: {label}")
            continue
        run_model(label, MODELS[label], data)

    print("\nDone.")


if __name__ == "__main__":
    main()
