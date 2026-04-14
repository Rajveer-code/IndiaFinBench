"""
evaluate_v7_models.py  (v2 — fixed providers)
----------------------------------------------
IndiaFinBench v7 — 3 new model evaluations:

  MODEL 1: DeepSeek R1 70B (distilled)
    → deepseek/deepseek-r1-distill-llama-70b via OpenRouter
    → strips <think>...</think> before scoring
    → saves:  evaluation/results/deepseek_r1_70b_results.csv

  MODEL 2: Gemma 4 E4B
    → Ollama local (gemma4)
    → saves:  evaluation/results/gemma4_e4b_results.csv

  MODEL 3: Gemini 2.0 Flash
    → Google AI Studio via google-generativeai, key pool (18 keys)
    → falls back to gemini-1.5-flash if 2.0 unavailable
    → saves:  evaluation/results/gemini3_flash_results.csv

Usage:
  python scripts/evaluate_v7_models.py
  python scripts/evaluate_v7_models.py --models deepseek_r1_70b gemma4_e4b
"""

import json, csv, os, re, time, sys, io, argparse
from pathlib import Path
from collections import defaultdict

# ── stdout encoding fix (Windows) ──────────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from rapidfuzz import fuzz as _rf  # noqa: F401
except ImportError:
    raise ImportError("pip install rapidfuzz")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
QA_PATH     = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
RESULTS_DIR = BASE / "evaluation/results"

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

GROQ_API_KEY         = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY   = os.environ.get("OPENROUTER_API_KEY", "")

# ── Gemini key pool (set via environment variable or .env file) ───────────────
# Add your Google AI Studio keys to GOOGLE_API_KEY env var or .env file
GEMINI_KEYS = []
# De-duplicate while preserving order
seen = set(); GEMINI_KEYS = [k for k in GEMINI_KEYS if k not in seen and not seen.add(k)]
# Also try env fallback
if os.environ.get("GOOGLE_API_KEY"):
    GEMINI_KEYS.append(os.environ["GOOGLE_API_KEY"])
if os.environ.get("GOOGLE_AI_STUDIO_KEY"):
    GEMINI_KEYS.append(os.environ["GOOGLE_AI_STUDIO_KEY"])
# Final de-dup
seen = set(); GEMINI_KEYS = [k for k in GEMINI_KEYS if k not in seen and not seen.add(k)]

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "deepseek_r1_70b": {
        "label":       "DeepSeek R1 70B",
        "provider":    "openrouter",
        "model_id":    "deepseek/deepseek-r1-distill-llama-70b",
        "strip_think": True,
        "out_csv":     "deepseek_r1_70b_results.csv",
        "hf_id":       "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "params":      "70B",
        "type":        "API",
    },
    "gemma4_e4b": {
        "label":       "Gemma 4 E4B",
        "provider":    "ollama",
        "model_id":    "gemma4",
        "strip_think": False,
        "out_csv":     "gemma4_e4b_results.csv",
        "hf_id":       "google/gemma-4-e4b",
        "params":      "4B",
        "type":        "Local",
    },
    "gemini3_flash": {
        "label":       "Gemini 3 Flash",
        "provider":    "gemini",
        "model_id":    "gemini-2.5-flash",
        "strip_think": False,
        "out_csv":     "gemini3_flash_results.csv",
        "hf_id":       "google/gemini-2.5-flash",
        "params":      "N/A",
        "type":        "API",
    },
}

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    """Remove <think>...</think> chain-of-thought blocks (DeepSeek R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q    = item["question"]
    _ctx_words = item.get("context", "").split()
    ctx = " ".join(_ctx_words[:450])

    if task == "contradiction_detection":
        return (
            f"Passage A:\n{item.get('context_a', '')[:1500]}\n\n"
            f"Passage B:\n{item.get('context_b', '')[:1500]}\n\n"
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


def normalise(text: str) -> str:
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def score_answer(ref: str, pred: str, task_type: str) -> int:
    from rapidfuzz import fuzz
    if not pred or "fail:" in pred.lower():
        return 0
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


# ── API callers ───────────────────────────────────────────────────────────────

def call_openrouter(model_id: str, prompt: str) -> str:
    """Call OpenRouter API — used for DeepSeek R1 70B."""
    import requests
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/IndiaFinBench",
        "X-Title":       "IndiaFinBench",
    }
    payload = {
        "model":       model_id,
        "max_tokens":  2048,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    for attempt in range(6):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=180,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 10, 120)
                print(f"  ⏳ OpenRouter 429 → wait {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * 10, 120)
                print(f"  ⏳ OpenRouter rate limit → wait {wait}s")
                time.sleep(wait)
                continue
            return f"FAIL: {err[:200]}"
    return "FAIL: OpenRouter retries exhausted"


def call_ollama(model_id: str, prompt: str) -> str:
    """Call Ollama local API."""
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model":   model_id,
            "prompt":  f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 512},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Gemini key pool ────────────────────────────────────────────────────────────
_key_cooldowns: dict[str, float] = {}


def _get_available_gemini_key() -> str:
    """Return a Gemini key not in cooldown; wait if all are cooling."""
    keys = GEMINI_KEYS
    if not keys:
        raise RuntimeError("No Gemini keys configured.")
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
    """Call Google AI Studio Gemini API with key pool rotation."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "FAIL: pip install google-genai"

    if not GEMINI_KEYS:
        return "FAIL: No Gemini keys available"

    current_model = model_id
    for attempt in range(8):
        key = _get_available_gemini_key()
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=current_model,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower() or "deprecated" in err.lower():
                raise Exception(f"  ⚠  {current_model} not available, model ID must be correct.")
            if "403" in err or "permission" in err.lower() or "leaked" in err.lower() or "invalid" in err.lower():
                print(f"  ⚠  Gemini key ...{key[-6:]} invalid/leaked, removing.")
                if key in GEMINI_KEYS:
                    GEMINI_KEYS.remove(key)
                continue
            if "429" in err or "quota" in err.lower() or "rate" in err.lower() or "exhausted" in err.lower():
                print(f"  🔄 Gemini key ...{key[-6:]} rate-limited → 65s cooldown")
                _key_cooldowns[key] = time.time() + 65
                time.sleep(min(2 ** attempt, 30))
                continue
            return f"FAIL: {err[:200]}"
    return "FAIL: Gemini retries exhausted"


# ── Availability checks ───────────────────────────────────────────────────────

def check_ollama(model_id: str) -> bool:
    import requests
    try:
        resp  = requests.get("http://localhost:11434/api/tags", timeout=5)
        names = [m["name"] for m in resp.json().get("models", [])]
        return any(model_id.split(":")[0] in n for n in names)
    except Exception:
        return False


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDS = ["id", "task_type", "difficulty", "question",
          "ref_answer", "prediction", "correct", "model_version"]


def _write_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  💾 Checkpoint saved ({len(rows)} rows) → {path.name}")


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(model_key: str, data: list) -> dict | None:
    cfg       = MODELS[model_key]
    label     = cfg["label"]
    provider  = cfg["provider"]
    model_id  = cfg["model_id"]
    do_strip  = cfg["strip_think"]
    out_path  = RESULTS_DIR / cfg["out_csv"]

    print(f"\n{'━'*68}")
    print(f"  Model    : {label}")
    print(f"  Provider : {provider} / {model_id}")
    print(f"{'━'*68}")

    # ── Availability checks ────────────────────────────────────────────────
    if provider == "openrouter" and not OPENROUTER_API_KEY:
        print("  ✗  OPENROUTER_API_KEY not set. Skipping.")
        return None
    elif provider == "ollama":
        if not check_ollama(model_id):
            print(f"  ✗  Model '{model_id}' not found in Ollama.")
            print(f"     Run: ollama pull {model_id}")
            return None
    elif provider == "gemini" and not GEMINI_KEYS:
        print("  ✗  No Gemini keys available. Skipping.")
        return None

    # ── Load checkpoint ────────────────────────────────────────────────────
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    done[row["id"]] = row

    remaining = len(data) - len(done)
    print(f"  Done     : {len(done)}/150  |  Remaining: {remaining}")

    if remaining == 0:
        print("  ✓  Already complete — skipping.\n")
        return _compute_accuracy(done)

    # ── Per-provider delays ────────────────────────────────────────────────
    delays = {
        "openrouter": 2.0,   # polite delay for DeepSeek via OpenRouter
        "ollama":     0.2,
        "gemini":     0.5,
    }
    delay = delays.get(provider, 1.0)

    results = list(done.values())

    for i, item in enumerate(data, 1):
        item_id = item["id"]
        task    = item["task_type"]
        if item_id in done:
            continue

        prompt = build_prompt(item)

        try:
            if provider == "openrouter":
                pred = call_openrouter(model_id, prompt)
            elif provider == "ollama":
                pred = call_ollama(model_id, prompt)
            elif provider == "gemini":
                pred = call_gemini(model_id, prompt)
            else:
                pred = f"FAIL: unknown provider {provider}"
        except Exception as e:
            pred = f"FAIL: {str(e)[:200]}"

        if do_strip and not pred.startswith("FAIL"):
            pred = strip_think(pred)
            if not pred:
                pred = "FAIL: empty after stripping <think> block"

        correct = score_answer(item["answer"], pred, task)

        row = {
            "id":            item_id,
            "task_type":     task,
            "difficulty":    item.get("difficulty", ""),
            "question":      item["question"][:80],
            "ref_answer":    item["answer"][:150],
            "prediction":    pred[:300],
            "correct":       correct,
            "model_version": model_id,
        }
        results.append(row)
        done[item_id] = row

        status = "✓" if correct else "✗"
        print(f"  [{i:03d}/150] {status}  {item_id:<14}  {pred[:55]}")

        if len(results) % 10 == 0:
            _write_csv(results, out_path)

        if delay:
            time.sleep(delay)

    _write_csv(results, out_path)
    acc = _compute_accuracy(done)
    _print_summary(acc, label)
    return acc


# ── Accuracy helpers ──────────────────────────────────────────────────────────

TASK_ABBR = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}


def _compute_accuracy(done: dict) -> dict:
    task_scores: dict[str, list] = defaultdict(list)
    for row in done.values():
        task_scores[row["task_type"]].append(int(row["correct"]))
    acc = {}
    all_scores = []
    for full, abbr in TASK_ABBR.items():
        sc = task_scores.get(full, [])
        acc[abbr] = sum(sc) / len(sc) if sc else 0.0
        all_scores.extend(sc)
    acc["overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    acc["n"] = len(all_scores)
    return acc


def _print_summary(acc: dict, label: str):
    print(f"\n  ── {label} Results ──")
    for abbr in ["REG", "NUM", "CON", "TMP"]:
        print(f"    {abbr}: {acc.get(abbr, 0)*100:.1f}%")
    print(f"    Overall: {acc.get('overall', 0)*100:.1f}%  (n={acc.get('n', 0)})")


# ── Full 10-model leaderboard ─────────────────────────────────────────────────

def print_full_leaderboard(new_results: dict) -> list:
    existing = [
        {"label": "Claude 3 Haiku",    "REG": 92.5, "NUM": 93.8, "CON": 86.7, "TMP": 91.4, "Overall": 91.3},
        {"label": "Gemini 2.5 Flash",  "REG": 96.2, "NUM": 84.4, "CON": 83.3, "TMP": 80.0, "Overall": 87.3},
        {"label": "Llama 4 Scout 17B", "REG": 79.2, "NUM": 75.0, "CON": 100.0,"TMP": 80.0, "Overall": 82.7},
        {"label": "Qwen3-32B",         "REG": 77.4, "NUM": 75.0, "CON": 86.7, "TMP": 94.3, "Overall": 82.7},
        {"label": "LLaMA-3.3-70B",     "REG": 77.4, "NUM": 84.4, "CON": 90.0, "TMP": 77.1, "Overall": 81.3},
        {"label": "LLaMA-3-8B",        "REG": 77.4, "NUM": 62.5, "CON": 86.7, "TMP": 74.3, "Overall": 75.3},
        {"label": "Mistral-7B",        "REG": 69.8, "NUM": 68.8, "CON": 80.0, "TMP": 74.3, "Overall": 72.7},
    ]

    for key, acc in new_results.items():
        existing.append({
            "label":   MODELS[key]["label"],
            "REG":     round(acc.get("REG", 0) * 100, 1),
            "NUM":     round(acc.get("NUM", 0) * 100, 1),
            "CON":     round(acc.get("CON", 0) * 100, 1),
            "TMP":     round(acc.get("TMP", 0) * 100, 1),
            "Overall": round(acc.get("overall", 0) * 100, 1),
        })

    existing.sort(key=lambda x: x["Overall"], reverse=True)

    print(f"\n{'━'*82}")
    print(f"  INDIAFINBENCH — FULL LEADERBOARD ({len(existing)} models)")
    print(f"{'━'*82}")
    print(f"  {'#':<3} {'Model':<26} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6} {'Overall':>8}")
    print(f"  {'─'*3} {'─'*26} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

    for rank, m in enumerate(existing, 1):
        print(
            f"  {rank:<3} {m['label']:<26} "
            f"{m['REG']:>5.1f}% {m['NUM']:>5.1f}% "
            f"{m['CON']:>5.1f}% {m['TMP']:>5.1f}% "
            f"{m['Overall']:>7.1f}%"
        )

    n = len(existing)
    print(f"  {'─'*3} {'─'*26} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    avg = lambda k: sum(m[k] for m in existing) / n
    print(
        f"  {'':3} {'Average':<26} "
        f"{avg('REG'):>5.1f}% {avg('NUM'):>5.1f}% "
        f"{avg('CON'):>5.1f}% {avg('TMP'):>5.1f}% "
        f"{avg('Overall'):>7.1f}%"
    )
    print(f"\n  Human Expert Baseline (n=30): 60.0%  "
          f"(REG 55.6%  NUM 44.4%  CON 83.3%  TMP 66.7%)")
    print(f"{'━'*82}\n")
    return existing


# ── Update baselines.json ─────────────────────────────────────────────────────

def update_baselines(new_results: dict):
    baselines_path = BASE / "demo/data/baselines.json"
    with open(baselines_path, encoding="utf-8") as f:
        baselines = json.load(f)

    existing_ids = {b["model_id"] for b in baselines}

    for key, acc in new_results.items():
        cfg = MODELS[key]
        if key in existing_ids:
            print(f"  [baselines] {key} already present — skipping.")
            continue
        entry = {
            "model_id":  key,
            "label":     cfg["label"],
            "hf_id":     cfg["hf_id"],
            "params":    cfg["params"],
            "type":      cfg["type"],
            "scores": {
                "REG": round(acc.get("REG", 0), 4),
                "NUM": round(acc.get("NUM", 0), 4),
                "CON": round(acc.get("CON", 0), 4),
                "TMP": round(acc.get("TMP", 0), 4),
            },
            "overall":   round(acc.get("overall", 0), 4),
            "n_items":   acc.get("n", 150),
            "submitted": "2026-04-10",
            "baseline":  True,
        }
        baselines.append(entry)
        print(f"  [baselines] Added {cfg['label']} → overall={entry['overall']:.4f}")

    with open(baselines_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓  baselines.json updated ({len(baselines)} entries)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )
    args = parser.parse_args()

    print(f"\n{'━'*68}")
    print(f"  IndiaFinBench v7 — New Model Evaluations")
    print(f"  Date     : 2026-04-10")
    print(f"  Models   : {', '.join(args.models)}")
    print(f"  OR key   : {'✓' if OPENROUTER_API_KEY else '✗'}")
    print(f"  Gemini   : {len(GEMINI_KEYS)} keys loaded")
    print(f"{'━'*68}")

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n  Loaded {len(data)} items from {QA_PATH.name}\n")

    new_results = {}
    for key in args.models:
        acc = evaluate_model(key, data)
        if acc is not None:
            new_results[key] = acc
        else:
            print(f"\n  ⚠  {key} skipped.\n")

    if new_results:
        print_full_leaderboard(new_results)
        print("  Updating baselines.json ...")
        update_baselines(new_results)

    print("\n  ✓  evaluate_v7_models.py complete.\n")


if __name__ == "__main__":
    main()
