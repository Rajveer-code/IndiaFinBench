"""
matched_budget_pilot.py
------------------------
Plan v3 cleanup item 6: pilot completion budgets {512, 1024, 2048} on a small
stratified subset, logging full responses, finish/stop reason, completion
tokens, and context fit, so a matched-budget re-run (Phase 2.2) uses a budget
chosen from evidence rather than assumed.

Reuses the exact model_id / provider pairs from evaluate.py, evaluate_new_models.py,
and evaluate_v7_models.py (Plan v3 F3 table) -- does not reimplement scoring or
prompt construction, only adds budget parameterisation and finish-reason logging,
which none of the original callers have.

Gemini 2.5 Flash / Gemini 2.5 Pro are excluded: no GEMINI_API_KEY / GOOGLE_API_KEY
or Vertex ADC credentials are available in this environment. Reported, not silently
skipped -- see the summary printed at the end.

Usage:
  python scripts/matched_budget_pilot.py                 # run the pilot (resumable)
  python scripts/matched_budget_pilot.py --summary-only   # just print the summary
"""

import json, os, re, time, random, argparse, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

BASE = Path(__file__).parent.parent
QA_PATH = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
OUT_PATH = BASE / "evaluation/matched_budget_pilot.json"
BUDGETS = [512, 1024, 2048]
N_PER_TASK = 8          # 8 REG + 8 NUM + 8 TMP = 24 items
SEED = 42
OLLAMA_NUM_CTX = 4096   # generation-side context window, set explicitly (not left at Ollama's default)

_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble."""

# model_id / provider pairs verified against evaluate.py, evaluate_new_models.py,
# evaluate_v7_models.py -- see Plan v3 F3 table. Gemini 2.5 Flash/Pro omitted (no creds).
MODELS = {
    "LLaMA-3-8B":          {"provider": "ollama",     "model_id": "llama3"},
    "Mistral-7B":          {"provider": "ollama",     "model_id": "mistral"},
    "Gemma 3 4B":          {"provider": "ollama",     "model_id": "gemma3:4b"},
    "LLaMA-3.3-70B":       {"provider": "groq",       "model_id": "llama-3.3-70b-versatile"},
    "Llama 4 Scout 17B":   {"provider": "groq",       "model_id": "meta-llama/llama-4-scout-17b-16e-instruct"},
    "Qwen3-32B":           {"provider": "groq",       "model_id": "qwen/qwen3-32b"},
    "GPT-OSS 120B":        {"provider": "groq",       "model_id": "openai/gpt-oss-120b"},
    "GPT-OSS 20B":         {"provider": "groq",       "model_id": "openai/gpt-oss-20b"},
    "Kimi K2":             {"provider": "groq",       "model_id": "moonshotai/kimi-k2-instruct"},
    "DeepSeek-R1-Distill": {"provider": "openrouter", "model_id": "deepseek/deepseek-r1-distill-llama-70b"},
}
EXCLUDED = {
    "Gemini 2.5 Flash": "no GEMINI_API_KEY / GOOGLE_API_KEY in this environment",
    "Gemini 2.5 Pro":   "no Vertex AI ADC credentials in this environment",
}


def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q = item["question"]
    ctx = " ".join(item.get("context", "").split()[:450])
    if task == "numerical_reasoning":
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nShow your calculation and give the final answer with units:"
    elif task == "temporal_reasoning":
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer precisely, noting relevant dates or sequences:"
    else:
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def sample_items():
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    by_task = {"regulatory_interpretation": [], "numerical_reasoning": [], "temporal_reasoning": []}
    for item in data:
        if item["task_type"] in by_task:
            by_task[item["task_type"]].append(item)
    rng = random.Random(SEED)
    sample = []
    for task, items in by_task.items():
        sample.extend(rng.sample(items, min(N_PER_TASK, len(items))))
    return sample


def call_groq(model_id: str, prompt: str, budget: int) -> dict:
    from groq import Groq
    if not GROQ_API_KEY:
        return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "GROQ_API_KEY not set"}
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id, max_tokens=budget, temperature=0.0,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            )
            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            return {
                "text": choice.message.content.strip() if choice.message.content else "",
                "finish_reason": choice.finish_reason,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "error": None,
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * 5, 60)
                time.sleep(wait)
                continue
            return {"text": "", "finish_reason": None, "completion_tokens": None, "error": err[:200]}
    return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "retries exhausted"}


def call_openrouter(model_id: str, prompt: str, budget: int) -> dict:
    import requests
    if not OPENROUTER_API_KEY:
        return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "OPENROUTER_API_KEY not set"}
    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer " + OPENROUTER_API_KEY, "Content-Type": "application/json"},
                json={"model": model_id, "max_tokens": budget, "temperature": 0.0,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]},
                timeout=90,
            )
            data = resp.json()
            if "error" in data:
                return {"text": "", "finish_reason": None, "completion_tokens": None, "error": str(data["error"])[:200]}
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return {
                "text": (choice["message"]["content"] or "").strip(),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "error": None,
            }
        except Exception as e:
            wait = min(2 ** attempt * 5, 60)
            time.sleep(wait)
    return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "retries exhausted"}


def call_ollama(model_id: str, prompt: str, budget: int) -> dict:
    import requests
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_id,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": budget, "num_ctx": OLLAMA_NUM_CTX},
            },
            timeout=180,
        )
        resp.raise_for_status()
        j = resp.json()
        return {
            "text": j.get("response", "").strip(),
            "finish_reason": j.get("done_reason"),
            "completion_tokens": j.get("eval_count"),
            "num_ctx": OLLAMA_NUM_CTX,
            "prompt_eval_count": j.get("prompt_eval_count"),
            "error": None,
        }
    except Exception as e:
        return {"text": "", "finish_reason": None, "completion_tokens": None, "error": str(e)[:200]}


CALLERS = {"groq": call_groq, "openrouter": call_openrouter, "ollama": call_ollama}


def is_truncated(result: dict, budget: int) -> bool:
    """Best-effort truncation flag, provider-agnostic."""
    fr = (result.get("finish_reason") or "").lower()
    if fr in ("length", "max_tokens"):
        return True
    ct = result.get("completion_tokens")
    if ct is not None and ct >= budget:
        return True
    return False


def run_pilot():
    sample = sample_items()
    print(f"Pilot sample: {len(sample)} items ({N_PER_TASK} each of REG/NUM/TMP), seed={SEED}")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Excluded (no credentials): {EXCLUDED}\n")

    results = []
    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("runs", [])
    done_keys = {(r["model"], r["item_id"], r["budget"]) for r in results}

    total = len(MODELS) * len(sample) * len(BUDGETS)
    done_n = len(done_keys)

    for model_label, cfg in MODELS.items():
        caller = CALLERS[cfg["provider"]]
        for budget in BUDGETS:
            for item in sample:
                key = (model_label, item["id"], budget)
                if key in done_keys:
                    continue
                prompt = build_prompt(item)
                out = caller(cfg["model_id"], prompt, budget)
                row = {
                    "model": model_label, "provider": cfg["provider"], "item_id": item["id"],
                    "task_type": item["task_type"], "budget": budget,
                    "response": out.get("text", ""), "finish_reason": out.get("finish_reason"),
                    "completion_tokens": out.get("completion_tokens"), "error": out.get("error"),
                    "truncated": is_truncated(out, budget),
                }
                if "num_ctx" in out:
                    row["num_ctx"] = out["num_ctx"]
                    row["prompt_eval_count"] = out.get("prompt_eval_count")
                results.append(row)
                done_n += 1
                status = "ERR" if out.get("error") else ("TRUNC" if row["truncated"] else "ok")
                print(f"  [{done_n:4d}/{total}] {model_label:<20} budget={budget:<5} {item['id']:<10} {status}")
                if done_n % 15 == 0:
                    OUT_PATH.write_text(json.dumps({"runs": results}, indent=2), encoding="utf-8")
                time.sleep(1.5 if cfg["provider"] != "ollama" else 0.2)

    OUT_PATH.write_text(json.dumps({"runs": results}, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} runs -> {OUT_PATH}")
    print_summary(results)


def print_summary(results=None):
    if results is None:
        if not OUT_PATH.exists():
            print("No pilot data yet. Run without --summary-only first.")
            return
        results = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("runs", [])

    from collections import defaultdict
    by_model_budget = defaultdict(list)
    for r in results:
        by_model_budget[(r["model"], r["budget"])].append(r)

    print(f"\n{'='*72}")
    print(f"{'Model':<20}{'Budget':>8}{'N':>6}{'Truncated':>12}{'Errors':>8}")
    print(f"{'-'*72}")
    per_budget_rate = defaultdict(list)
    for model in MODELS:
        for budget in BUDGETS:
            rows = by_model_budget.get((model, budget), [])
            n = len(rows)
            if n == 0:
                print(f"{model:<20}{budget:>8}{n:>6}{'--':>12}{'--':>8}")
                continue
            trunc = sum(1 for r in rows if r["truncated"])
            err = sum(1 for r in rows if r.get("error"))
            rate = trunc / n
            per_budget_rate[budget].append(rate)
            print(f"{model:<20}{budget:>8}{n:>6}{f'{trunc}/{n} ({rate:.0%})':>12}{err:>8}")
    print(f"{'-'*72}")
    for budget in BUDGETS:
        rates = per_budget_rate.get(budget, [])
        mean_rate = sum(rates) / len(rates) if rates else float("nan")
        print(f"  mean truncation rate at budget={budget}: {mean_rate:.1%} (across {len(rates)} models)")

    print(f"\nExcluded from this pilot (no credentials): {EXCLUDED}")
    print("Recommendation: the lowest budget in BUDGETS whose mean truncation rate is")
    print("within a small margin of the next higher budget's rate -- read this table,")
    print("do not auto-select, since 'materially increasing' is a judgment call.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        print_summary()
    else:
        run_pilot()
