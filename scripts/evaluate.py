"""
evaluate.py  (final version)
-----------------------------
IndiaFinBench — Phase 3 Model Evaluation

Models:
  llama3   — LLaMA-3-8B        (Ollama, local, free)
  mistral  — Mistral-7B        (Ollama, local, free)
  haiku    — Claude 3 Haiku    (Anthropic API)
  gemini   — Gemini 2.5 Flash  (Google AI Studio, key pool)
  groq70b  — LLaMA-3.3-70B    (Groq API, free)

Usage:
  python scripts/evaluate.py --models gemini
  python scripts/evaluate.py --models groq70b
  python scripts/evaluate.py --models llama3 mistral haiku gemini groq70b
"""

import json, csv, os, time, re, argparse

# BUG 4 FIX: Hard import — rapidfuzz is required; silent fallback would silently
# change all accuracy numbers if the package were missing.
try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz  # noqa: F401
except ImportError:
    raise ImportError("rapidfuzz required: pip install rapidfuzz")

# ══════════════════════════════════════════════════════════════════
#  EDIT THIS SECTION — paste your keys here
# ══════════════════════════════════════════════════════════════════

GROQ_KEY = ""  # Set via: export GROQ_API_KEY=your_key

GEMINI_KEYS = []  # Set via: export GOOGLE_API_KEY=your_key

ANTHROPIC_KEY = ""  # Set via: export ANTHROPIC_API_KEY=your_key

# ══════════════════════════════════════════════════════════════════

QA_PATH     = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
RESULTS_DIR = "evaluation/results"

MODELS = {
    "llama3": {
        "label":    "LLaMA-3-8B",
        "provider": "ollama",
        "model_id": "llama3",
        "version":  "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "mistral": {
        "label":    "Mistral-7B",
        "provider": "ollama",
        "model_id": "mistral",
        "version":  "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "haiku": {
        "label":    "Claude 3 Haiku",
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",  # BUG 1 FIX: was claude-haiku-4-5-20251001
        "version":  "claude-3-haiku-20240307",
    },
    "gemini": {
        "label":    "Gemini 2.5 Flash",
        "provider": "gemini_pool",
        "model_id": "gemini-2.5-flash",
        "version":  "gemini-2.5-flash",
    },
    "groq70b": {
        "label":    "LLaMA-3.3-70B (Groq)",
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "version":  "meta-llama/Llama-3.3-70B-Instruct",
    },
}

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble."""


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q    = item["question"]
    # BUG 5 FIX: word-boundary truncation instead of mid-word char slice
    _ctx_words = item.get("context", "").split()
    ctx = " ".join(_ctx_words[:450])  # ~3000 chars, word-boundary safe

    if task == "contradiction_detection":
        return (
            f"Passage A:\n{item.get('context_a','')[:1500]}\n\n"
            f"Passage B:\n{item.get('context_b','')[:1500]}\n\n"
            f"Question: {q}\n\n"
            f"Answer with 'Yes' or 'No' then one sentence of explanation:"
        )
    elif task == "numerical_reasoning":
        return (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Show your calculation and give the final answer with units:"
        )
    elif task == "temporal_reasoning":
        return (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Answer precisely, noting relevant dates or sequences:"
        )
    else:
        return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


# ── Gemini key pool ────────────────────────────────────────────────────────────

_key_cooldowns = {}   # key -> cooldown_until timestamp

def _get_available_key():
    """Returns a Gemini key not in cooldown. Waits if all cooling."""
    keys = [k for k in GEMINI_KEYS if k]
    if not keys:
        env_key = os.environ.get("GOOGLE_API_KEY", "")
        if env_key:
            keys = [env_key]
    if not keys:
        raise RuntimeError(
            "No Gemini keys configured.\n"
            "Add keys to the GEMINI_KEYS list at the top of evaluate.py\n"
            "OR run: export GOOGLE_API_KEY=your_key"
        )
    while True:
        now = time.time()
        for k in keys:
            if now >= _key_cooldowns.get(k, 0):
                return k
        soonest = min(_key_cooldowns.get(k, 0) for k in keys)
        wait = max(1, soonest - time.time() + 1)
        print(f"  ⏳ All Gemini keys in cooldown — waiting {wait:.0f}s ...")
        time.sleep(wait)


def call_gemini(model_id: str, prompt: str) -> str:
    from google import genai

    for attempt in range(8):
        key = _get_available_key()
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=model_id,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 200,
                    "thinking_config": {"thinking_budget": 0},
                },
            )
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower() or "exhausted" in err.lower():
                print(f"  🔄 Gemini key ...{key[-6:]} rate-limited → 65s cooldown")
                _key_cooldowns[key] = time.time() + 65
                time.sleep(min(2 ** attempt, 30))
                continue
            return f"FAIL: {err[:100]}"

    return "FAIL: all Gemini retries exhausted"


# ── Groq caller ────────────────────────────────────────────────────────────────

def call_groq(model_id: str, prompt: str, client) -> str:
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=200,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * 5, 60)
                print(f"  ⏳ Groq rate limit → wait {wait}s")
                time.sleep(wait)
                continue
            return f"FAIL: {err[:100]}"
    return "FAIL: Groq retries exhausted"


# ── Ollama caller ──────────────────────────────────────────────────────────────

def call_ollama(model_id: str, prompt: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":  model_id,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 300},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Anthropic caller ───────────────────────────────────────────────────────────

def call_anthropic_api(model_id: str, prompt: str, client) -> str:
    resp = client.messages.create(
        model=model_id,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Client init ────────────────────────────────────────────────────────────────

def init_client(provider: str):
    if provider == "ollama":
        import requests
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
            return "ollama_ok"
        except Exception:
            print("  ✗  Ollama not running. Run: ollama serve")
            return None

    elif provider == "anthropic":
        import anthropic
        key = ANTHROPIC_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            print("  ✗  ANTHROPIC_API_KEY not set")
            return None
        return anthropic.Anthropic(api_key=key)

    elif provider == "gemini_pool":
        try:
            from google import genai  # noqa
        except ImportError:
            print("  ✗  Run: pip install google-genai")
            return None
        keys = [k for k in GEMINI_KEYS if k]
        if not keys:
            env_key = os.environ.get("GOOGLE_API_KEY", "")
            if env_key:
                keys = [env_key]
        if not keys:
            print("  ✗  No Gemini keys. Add to GEMINI_KEYS list at top of script.")
            return None
        print(f"  Gemini key pool: {len(keys)} key(s) loaded")
        return "gemini_pool_ok"

    elif provider == "groq":
        try:
            from groq import Groq
        except ImportError:
            print("  ✗  Run: pip install groq")
            return None
        key = GROQ_KEY or os.environ.get("GROQ_API_KEY", "")
        if not key:
            print("  ✗  GROQ_API_KEY not set")
            return None
        return Groq(api_key=key)

    return None


# ── Scoring ────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def score_answer(ref: str, pred: str, task_type: str) -> int:
    if not pred or "fail:" in pred.lower() or "cannot be determined" in pred.lower():
        return 0
    # BUG 4 FIX: rapidfuzz is imported at module level (hard fail); no silent fallback.
    from rapidfuzz import fuzz

    if task_type == "contradiction_detection":
        r = normalise(ref)
        p = normalise(pred)
        ref_yn = "yes" if r.startswith("yes") else ("no" if r.startswith("no") else r)
        ann_yn = "yes" if p.startswith("yes") else ("no" if p.startswith("no") else p)
        return int(ref_yn == ann_yn)

    rn, pn = normalise(ref), normalise(pred)
    nums_r = set(re.findall(r"\d[\d,]*\.?\d*", rn))
    nums_p = set(re.findall(r"\d[\d,]*\.?\d*", pn))
    if nums_r and nums_p and nums_r == nums_p:
        return 1
    return int(fuzz.token_set_ratio(rn, pn) / 100.0 >= 0.72)


# ── Core evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(model_key: str, data: list):
    cfg      = MODELS[model_key]
    label    = cfg["label"]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    out_path = os.path.join(RESULTS_DIR, f"{model_key}_results.csv")

    delays = {"ollama": 0.0, "anthropic": 0.3,
              "gemini_pool": 0.5, "groq": 0.5}
    delay = delays.get(provider, 1.0)

    # Load checkpoint
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    done[row["id"]] = row

    remaining = len(data) - len(done)
    print(f"\n{'━'*65}")
    print(f"  Model    : {label}")
    print(f"  Done     : {len(done)}/150  |  Remaining: {remaining}")
    print(f"{'━'*65}\n")

    if remaining == 0:
        print("  Already complete — skipping.\n")
        return

    client = init_client(provider)
    if client is None:
        print(f"  Skipping {label}.\n")
        return

    results = list(done.values())
    errors  = 0
    task_sc = {t: [] for t in ["regulatory_interpretation",
               "numerical_reasoning", "contradiction_detection",
               "temporal_reasoning"]}
    for row in results:
        t = row.get("task_type", "")
        if t in task_sc:
            task_sc[t].append(int(row.get("correct", 0)))

    for i, item in enumerate(data, 1):
        iid  = item["id"]
        task = item["task_type"]
        if iid in done:
            continue

        prompt = build_prompt(item)

        try:
            if provider == "gemini_pool":
                pred = call_gemini(model_id, prompt)
            elif provider == "groq":
                pred = call_groq(model_id, prompt, client)
            elif provider == "ollama":
                pred = call_ollama(model_id, prompt)
            elif provider == "anthropic":
                pred = call_anthropic_api(model_id, prompt, client)
            else:
                pred = "FAIL: unknown provider"
        except Exception as e:
            pred = f"FAIL: {str(e)[:80]}"

        correct = score_answer(item["answer"], pred, task)
        if pred.startswith("FAIL"):
            correct = 0
            errors += 1

        task_sc[task].append(correct)
        results.append({
            "id":           iid,
            "task_type":    task,
            "difficulty":   item.get("difficulty", ""),
            "question":     item["question"][:80],
            "ref_answer":   item["answer"][:100],
            "prediction":   pred[:200],
            "correct":      correct,
            "model_version": cfg.get("version", model_id),  # BUG 1 FIX: log exact version
        })

        print(f"  [{i:03d}/150]  {iid:<12}  {'✓' if correct else '✗'}")

        if i % 25 == 0:
            _save(results, out_path)
            n_done = sum(len(v) for v in task_sc.values())
            n_corr = sum(sum(v) for v in task_sc.values())
            acc    = n_corr / n_done * 100 if n_done else 0
            print(f"\n  💾  Checkpoint {i}/150 | Acc: {acc:.1f}% | Errors: {errors}\n")

        time.sleep(delay)

    _save(results, out_path)

    print(f"\n  ── Results: {label} ──")
    print(f"  {'Task':<32}  {'N':>4}  {'✓':>5}  {'Acc':>6}")
    print(f"  {'─'*32}  {'─'*4}  {'─'*5}  {'─'*6}")
    tot_n = tot_c = 0
    for task, sc in task_sc.items():
        if sc:
            n, c = len(sc), sum(sc)
            tot_n += n; tot_c += c
            print(f"  {task:<32}  {n:>4}  {c:>5}  {c/n*100:>5.1f}%")
    if tot_n:
        print(f"  {'─'*32}  {'─'*4}  {'─'*5}  {'─'*6}")
        print(f"  {'OVERALL':<32}  {tot_n:>4}  {tot_c:>5}  {tot_c/tot_n*100:>5.1f}%")
    print(f"  Errors: {errors}\n")


def _save(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["id","task_type","difficulty","question",
              "ref_answer","prediction","correct","model_version"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ── Leaderboard ────────────────────────────────────────────────────────────────

def print_leaderboard(models_to_run):
    task_keys = ["regulatory_interpretation", "numerical_reasoning",
                 "contradiction_detection",   "temporal_reasoning"]
    print(f"\n{'━'*72}")
    print(f"  INDIAFINBENCH LEADERBOARD")
    print(f"{'━'*72}")
    print(f"  {'Model':<24}  {'REG':>6}  {'NUM':>6}  {'CON':>6}  {'TMP':>6}  {'Overall':>8}")
    print(f"  {'─'*24}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")

    for mk in models_to_run:
        path = os.path.join(RESULTS_DIR, f"{mk}_results.csv")
        if not os.path.exists(path):
            continue
        rows  = list(csv.DictReader(open(path, encoding="utf-8")))
        valid = [r for r in rows if "FAIL" not in r.get("prediction","")]
        if not valid:
            continue
        cols = []
        for task in task_keys:
            t_rows = [r for r in valid if r["task_type"]==task]
            if t_rows:
                a = sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100
                cols.append(f"{a:>5.1f}%")
            else:
                cols.append("   N/A")
        ov = sum(int(r["correct"]) for r in valid)/len(valid)*100
        print(f"  {MODELS[mk]['label']:<24}  {'  '.join(cols)}  {ov:>7.1f}%")

    print(f"{'━'*72}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=list(MODELS.keys()) + ["all"])
    args = parser.parse_args()

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    to_run = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"\n{'━'*65}")
    print(f"  IndiaFinBench — Phase 3 Evaluation")
    print(f"  Items  : {len(data)}")
    print(f"  Models : {', '.join(MODELS[m]['label'] for m in to_run)}")
    print(f"{'━'*65}")

    for mk in to_run:
        evaluate_model(mk, data)

    print_leaderboard(to_run)


if __name__ == "__main__":
    main()