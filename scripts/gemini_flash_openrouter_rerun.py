"""
gemini_flash_openrouter_rerun.py
------------------------------
Cleanup pass, 2026-09-04: Gemini 2.5 Flash's direct Google AI Studio matched-budget rerun
stalled for hours on a persistent quota block, even with 5-key rotation. Author authorized
spending the remaining ~$0.35 OpenRouter balance to run it via OpenRouter instead
(google/gemini-2.5-flash, Google AI Studio Flex tier only) as a SEPARATE, self-contained
script rather than folding into matched_budget_pilot.py's shared CALLERS -- this run needs
its own request shape (provider.only, provider.max_price, service_tier, reasoning disabled)
and its own cost accounting, and keeping it isolated makes the one-off nature auditable.

Endpoint/pricing verified live 2026-09-04 against
https://openrouter.ai/api/v1/models/google/gemini-2.5-flash/endpoints:
  google-ai-studio/flex: prompt $0.15/M, completion $1.25/M (matches author's own research).

Mandatory two-step protocol (per author's explicit instruction -- do not skip):
  1. --probe: run N items (default 8), log full usage/cost/finish_reason/provider/service_tier
     per item, print the extrapolated cost for all 406, and STOP without writing model output.
  2. --full: only after the probe's extrapolation is confirmed safe, run all 406 items fresh
     (the existing partial Google-direct file was moved aside, not appended to -- provider is
     a confound the matched-budget experiment cannot absorb) and write real output.

Every request is pinned to provider.only=["google-ai-studio"] + service_tier="flex" +
provider.max_price -- per OpenRouter's own docs, flex service_tier restricts routing to flex
endpoints and never silently falls back to a costlier default-tier endpoint; it surfaces a
capacity error instead. reasoning.max_tokens=0 disables Gemini's hidden-thinking mode (the
same failure class that silently burned budget on Qwen3-32B and, candidately, DeepSeek-R1-
Distill earlier in this project) so it can't consume the 512-token budget invisibly.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
import truststore
truststore.inject_into_ssl()
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.matched_budget_pilot import SYSTEM_PROMPT  # noqa: E402 -- reuse, don't reimplement

BASE = Path(__file__).parent.parent
QA_PATH = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
OUT_PATH = BASE / "evaluation/results_matched/gemini_25_flash_results.csv"
MANIFEST_PATH = BASE / "evaluation/gemini_flash_openrouter_manifest.json"
BUDGET = 512
MODEL_ID = "google/gemini-2.5-flash"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
FIELDS = ["id", "task_type", "difficulty", "question", "ref_answer", "prediction",
          "finish_reason", "completion_tokens", "error", "model_version"]


def build_prompt(item: dict) -> str:
    task = item["task_type"]
    ctx = item.get("context", "")
    q = item["question"]
    if task == "contradiction_detection":
        pa = item.get("passage_a", ctx)
        pb = item.get("passage_b", "")
        return (f"Passage A: {pa}\n\nPassage B: {pb}\n\nQuestion: {q}\n\n"
                f"Answer with 'Yes' or 'No' followed by a one-sentence explanation.")
    return f"Context: {ctx}\n\nQuestion: {q}"


def call(prompt: str) -> dict:
    """One OpenRouter request, pinned to Google AI Studio Flex, reasoning disabled.
    Returns the full parsed response plus timing; caller decides what to log/keep."""
    body = {
        "model": MODEL_ID,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
        "max_tokens": BUDGET,
        "temperature": 0.0,
        "provider": {"only": ["google-ai-studio"], "max_price": {"prompt": 0.15, "completion": 1.25}},
        "service_tier": "flex",
        "reasoning": {"max_tokens": 0},
        "usage": {"include": True},
    }
    t0 = time.time()
    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json=body, timeout=120,
            )
            data = resp.json()
            latency = time.time() - t0
            if "error" in data:
                err = str(data["error"])
                if "429" in err or "capacity" in err.lower() or "rate" in err.lower():
                    time.sleep(min(2 ** attempt * 5, 60))
                    continue
                return {"error": err[:300], "raw": data, "latency": latency}
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return {
                "text": (choice["message"].get("content") or "").strip(),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": usage.get("cost"),
                "provider": data.get("provider"),
                "service_tier": data.get("service_tier") or (data.get("usage") or {}).get("service_tier"),
                "latency": latency,
                "error": None,
                "raw": data,
            }
        except Exception as e:
            time.sleep(min(2 ** attempt * 5, 60))
    return {"error": "retries exhausted", "latency": time.time() - t0}


def run_probe(n: int):
    items = json.loads(QA_PATH.read_text(encoding="utf-8"))[:n]
    total_cost = 0.0
    results = []
    print(f"=== COST PROBE: {n} items, model={MODEL_ID}, tier=flex, provider=google-ai-studio only ===\n")
    for it in items:
        prompt = build_prompt(it)
        r = call(prompt)
        results.append({"id": it["id"], **{k: v for k, v in r.items() if k != "raw"}})
        cost = r.get("cost")
        if cost is not None:
            total_cost += cost
        print(f"[{it['id']}] finish={r.get('finish_reason')} "
              f"prompt_tok={r.get('prompt_tokens')} compl_tok={r.get('completion_tokens')} "
              f"reasoning_tok={r.get('reasoning_tokens')} cost=${cost} "
              f"provider={r.get('provider')} tier={r.get('service_tier')} "
              f"latency={r.get('latency', 0):.1f}s error={r.get('error')}")

    n_ok = sum(1 for r in results if not r.get("error"))
    print(f"\n{n_ok}/{n} succeeded.")
    if total_cost > 0 and n_ok > 0:
        per_item = total_cost / n_ok
        projected_406 = per_item * 406
        print(f"Actual cost this probe: ${total_cost:.5f} ({n_ok} items, ${per_item:.5f}/item)")
        print(f"Projected cost for all 406: ${projected_406:.4f}")
    else:
        print("No cost data returned (cost field empty) -- cannot extrapolate reliably; "
              "compute from prompt/completion tokens manually before proceeding.")
    Path("build").mkdir(exist_ok=True)
    Path("build/gemini_openrouter_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nFull probe detail -> build/gemini_openrouter_probe.json")


def run_full():
    items = json.loads(QA_PATH.read_text(encoding="utf-8"))
    assert len(items) == 406, f"expected 406 items, got {len(items)}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"REFUSING to overwrite existing {OUT_PATH} -- delete it first if you mean to restart.")
        sys.exit(1)

    total_cost = 0.0
    n_err = 0
    provider_seen, tier_seen = set(), set()
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, it in enumerate(items, 1):
            prompt = build_prompt(it)
            r = call(prompt)
            if r.get("error"):
                n_err += 1
                row = {"id": it["id"], "task_type": it["task_type"], "difficulty": it.get("difficulty", ""),
                       "question": it["question"], "ref_answer": it.get("answer", ""), "prediction": "",
                       "finish_reason": "", "completion_tokens": "", "error": r["error"],
                       "model_version": MODEL_ID}
            else:
                total_cost += r.get("cost") or 0.0
                provider_seen.add(r.get("provider"))
                tier_seen.add(r.get("service_tier"))
                row = {"id": it["id"], "task_type": it["task_type"], "difficulty": it.get("difficulty", ""),
                       "question": it["question"], "ref_answer": it.get("answer", ""),
                       "prediction": r["text"], "finish_reason": r.get("finish_reason"),
                       "completion_tokens": r.get("completion_tokens"), "error": "",
                       "model_version": MODEL_ID}
            writer.writerow(row)
            f.flush()
            if i % 20 == 0 or i == len(items):
                print(f"[{i}/406] total_cost=${total_cost:.4f} errors={n_err}")

    manifest = {
        "model_id": MODEL_ID, "provider_restriction": "google-ai-studio",
        "service_tier_requested": "flex", "providers_actually_used": sorted(p for p in provider_seen if p),
        "service_tiers_actually_used": sorted(t for t in tier_seen if t),
        "budget": BUDGET, "n_items": len(items), "n_errors": n_err,
        "total_cost_usd": round(total_cost, 5),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(items) - n_err}/{len(items)} succeeded, ${total_cost:.4f} spent.")
    print(f"Manifest -> {MANIFEST_PATH}")
    if n_err:
        print(f"{n_err} rows errored -- rerun with a fresh invocation to retry (file is NOT overwritten "
              f"automatically; delete {OUT_PATH} to allow a restart, or write a small retry pass).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, nargs="?", const=8, default=None,
                     help="Run a cost probe on N items (default 8) and stop.")
    ap.add_argument("--full", action="store_true", help="Run all 406 items and write real output.")
    args = ap.parse_args()
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not set in .env"); sys.exit(1)
    if args.probe is not None:
        run_probe(args.probe)
    elif args.full:
        run_full()
    else:
        print("Pass --probe [N] first, then --full once the projection looks safe.")
