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

2026-09-03 update: fresh Groq/OpenRouter/Gemini keys supplied. Groq deprecated 4 of the
6 models this benchmark originally used there (llama-3.3-70b-versatile, Llama 4 Scout,
Qwen3-32B, Kimi K2 all 404 now -- confirmed via a live models.list() call, only the two
GPT-OSS models survive on Groq); moved those 4 to OpenRouter, where their open-weight
checkpoints are still hosted (verified live). Qwen3-32B on OpenRouter's default route
(Nebius) answers in thinking mode by default, which burns the budget on invisible
reasoning tokens before any visible answer -- suppressed via the documented "/no_think"
directive, matching the original benchmark's direct-answer, no-preamble system prompt.
Gemini 2.5 Flash/Pro added via Google AI Studio (google-genai SDK).

Usage:
  python scripts/matched_budget_pilot.py                 # run the pilot (resumable)
  python scripts/matched_budget_pilot.py --summary-only   # just print the summary
"""

import json, os, re, time, random, argparse, sys, io
from pathlib import Path

import truststore
truststore.inject_into_ssl()  # Avast's HTTPS-scanning AV intercepts TLS otherwise (documented
# earlier in this project); without this, Groq/OpenRouter/Gemini calls fail with a raw
# "Connection error" (SSL: CERTIFICATE_VERIFY_FAILED) that looks like a network/key problem.

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
# GEMINI_API_KEYS holds every key we have, comma-separated; call_gemini rotates across them on
# 429/quota instead of sleeping on an exhausted key (2026-09-03: rotation added after the single
# original key throttled for 2+ hours with no recovery -- multiple fresh keys fixes this).
GEMINI_API_KEYS = [k.strip() for k in
                    os.environ.get("GEMINI_API_KEYS", os.environ.get("GOOGLE_API_KEY", "")).split(",")
                    if k.strip()]
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""  # back-compat for any other importer
_gemini_key_idx = [0]  # mutable cell so call_gemini can advance it across calls

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble."""

# model_id / provider pairs verified live 2026-09-03. Groq's own models.list() no longer
# includes llama-3.3-70b-versatile / llama-4-scout / qwen3-32b / kimi-k2-instruct -- moved
# to OpenRouter, confirmed reachable there under these exact IDs. Qwen3-32B gets a
# prompt_suffix to suppress OpenRouter/Nebius's default thinking mode (see module docstring).
MODELS = {
    "LLaMA-3-8B":          {"provider": "ollama",     "model_id": "llama3"},
    "Mistral-7B":          {"provider": "ollama",     "model_id": "mistral"},
    "Gemma 3 4B":          {"provider": "ollama",     "model_id": "gemma3:4b"},
    "GPT-OSS 120B":        {"provider": "groq",       "model_id": "openai/gpt-oss-120b"},
    "GPT-OSS 20B":         {"provider": "groq",       "model_id": "openai/gpt-oss-20b"},
    "DeepSeek-R1-Distill": {"provider": "openrouter", "model_id": "deepseek/deepseek-r1-distill-llama-70b"},
    "LLaMA-3.3-70B":       {"provider": "openrouter", "model_id": "meta-llama/llama-3.3-70b-instruct"},
    "Llama 4 Scout 17B":   {"provider": "openrouter", "model_id": "meta-llama/llama-4-scout"},
    "Kimi K2":             {"provider": "openrouter", "model_id": "moonshotai/kimi-k2"},
    "Qwen3-32B":           {"provider": "openrouter", "model_id": "qwen/qwen3-32b", "prompt_suffix": " /no_think"},
    "Gemini 2.5 Flash":    {"provider": "gemini",     "model_id": "gemini-2.5-flash"},
}
# Gemini 2.5 Pro: confirmed unreachable with this fresh key -- live API error is explicit:
# "This model models/gemini-2.5-pro is no longer available to new users. Please update your
# code to use models/gemini-3.1-pro-preview." A different model generation is not a matched
# re-run of the same checkpoint, so it is excluded rather than substituted.
EXCLUDED = {
    "Gemini 2.5 Pro": "no longer available to new AI Studio API keys (platform restriction, "
                       "confirmed via live 404 from the Gemini API itself, not a credentials problem)",
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


def call_gemini(model_id: str, prompt: str, budget: int) -> dict:
    if not GEMINI_API_KEYS:
        return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "GEMINI_API_KEYS/GOOGLE_API_KEY not set"}
    from google import genai
    n_keys = len(GEMINI_API_KEYS)
    # Try every key at most twice before falling back to a real sleep -- a 429 on one key
    # rotates to the next key immediately (no sleep), since a fresh key's quota is independent.
    for attempt in range(n_keys * 2):
        key = GEMINI_API_KEYS[_gemini_key_idx[0] % n_keys]
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model_id,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config={
                    "temperature": 0.0,
                    "max_output_tokens": budget,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            cand = resp.candidates[0] if resp.candidates else None
            usage = getattr(resp, "usage_metadata", None)
            return {
                "text": (resp.text or "").strip() if resp.text else "",
                "finish_reason": getattr(cand, "finish_reason", None).name if cand and getattr(cand, "finish_reason", None) else None,
                "completion_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                "error": None,
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower() or "exhausted" in err.lower():
                _gemini_key_idx[0] += 1
                if (attempt + 1) % n_keys == 0:
                    # every key just failed once this round -- back off before the next round
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err)
                    time.sleep(float(m.group(1)) + 3 if m else 15)
                continue
            if "404" in err and "no longer available" in err.lower():
                # this key's account tier doesn't have this model at all (not a quota issue) --
                # rotate past it permanently for this run rather than retrying it.
                _gemini_key_idx[0] += 1
                continue
            return {"text": "", "finish_reason": None, "completion_tokens": None, "error": err[:200]}
    return {"text": "", "finish_reason": None, "completion_tokens": None, "error": "all keys exhausted"}


CALLERS = {"groq": call_groq, "openrouter": call_openrouter, "ollama": call_ollama, "gemini": call_gemini}


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
    # Errored rows do NOT count as done -- a resume should retry them, not freeze a transient
    # failure into the record forever. Drop them from both the in-memory results and done_keys.
    results = [r for r in results if not r.get("error")]
    done_keys = {(r["model"], r["item_id"], r["budget"]) for r in results}

    total = len(MODELS) * len(sample) * len(BUDGETS)
    done_n = len(done_keys)

    delay_map = {"ollama": 0.2, "groq": 1.5, "openrouter": 2.0, "gemini": 8.0}
    # Gemini free tier: confirmed live 429 at ~20 req/min for gemini-2.5-flash. 8s spacing
    # keeps us near 7-8 RPM, with headroom for the retry loop's own calls.

    for model_label, cfg in MODELS.items():
        caller = CALLERS[cfg["provider"]]
        for budget in BUDGETS:
            for item in sample:
                key = (model_label, item["id"], budget)
                if key in done_keys:
                    continue
                prompt = build_prompt(item) + cfg.get("prompt_suffix", "")
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
                time.sleep(delay_map.get(cfg["provider"], 2.0))

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
