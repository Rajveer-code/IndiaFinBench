"""
evaluate_new_models_v2.py
--------------------------
IndiaFinBench — 3 additional free models

  gpt_oss_120b  — GPT-OSS 120B     (Cerebras, FREE, no credit card)
  qwen3_235b    — Qwen3-235B       (Cerebras, FREE, no credit card)
  kimi_k2       — Kimi K2          (Groq, FREE, you already have key)

Usage:
  python scripts/evaluate_new_models_v2.py --models gpt_oss_120b
  python scripts/evaluate_new_models_v2.py --models qwen3_235b
  python scripts/evaluate_new_models_v2.py --models kimi_k2
  python scripts/evaluate_new_models_v2.py --models gpt_oss_120b qwen3_235b kimi_k2

Setup:
  pip install cerebras-cloud-sdk groq rapidfuzz
  Set CEREBRAS_API_KEY and GROQ_API_KEY in your .env file
"""

import json, csv, os, re, time, sys, io, argparse
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("pip install rapidfuzz")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
QA_PATH     = BASE / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
RESULTS_DIR = BASE / "evaluation/results"

# ── Load .env ──────────────────────────────────────────────────────────────────
_env_path = BASE / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "gpt_oss_120b": {
        "label":    "GPT-OSS 120B",
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "out_csv":  "gpt_oss_120b_results.csv",
        "hf_id":    "openai/gpt-oss-120b",
        "params":   "120B",
        "type":     "API",
    },
    "gpt_oss_20b": {
        "label":    "GPT-OSS 20B",
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "out_csv":  "gpt_oss_20b_results.csv",
        "hf_id":    "openai/gpt-oss-20b",
        "params":   "20B",
        "type":     "API",
    },
    "kimi_k2": {
        "label":    "Kimi K2",
        "provider": "groq",
        "model_id": "moonshotai/kimi-k2-instruct",
        "out_csv":  "kimi_k2_results.csv",
        "hf_id":    "moonshotai/Kimi-K2-Instruct",
        "params":   "1T (32B active)",
        "type":     "API",
    },
    "nemotron_120b": {
        "label":    "NVIDIA Nemotron 120B",
        "provider": "openrouter",
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "out_csv":  "nemotron_120b_results.csv",
        "hf_id":    "nvidia/Nemotron-3-Super-120B-A12B",
        "params":   "120B",
        "type":     "API",
    },
    "qwen3_235b": {
        "label":    "Qwen3-235B",
        "provider": "cerebras",
        "model_id": "qwen-3-235b-a22b-instruct-2507",
        "out_csv":  "qwen3_235b_results.csv",
        "hf_id":    "Qwen/Qwen3-235B-A22B",
        "params":   "235B (22B active)",
        "type":     "API",
    },
}

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble."""

FIELDS = ["id", "task_type", "difficulty", "question",
          "ref_answer", "prediction", "correct", "model_version"]

TASK_ABBR = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q    = item["question"]
    ctx  = " ".join(item.get("context", "").split()[:450])

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


# ── Scoring ────────────────────────────────────────────────────────────────────

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


# ── API callers ────────────────────────────────────────────────────────────────

def call_cerebras(model_id: str, prompt: str) -> str:
    """Call Cerebras API — OpenAI-compatible, free tier."""
    try:
        from cerebras.cloud.sdk import Cerebras
    except ImportError:
        return "FAIL: pip install cerebras-cloud-sdk"

    if not CEREBRAS_API_KEY:
        return "FAIL: CEREBRAS_API_KEY not set in .env"

    client = Cerebras(api_key=CEREBRAS_API_KEY)

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=512,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower() or "limit" in err.lower():
                wait = min(2 ** attempt * 10, 120)
                print(f"  ⏳ Cerebras rate limit → wait {wait}s")
                time.sleep(wait)
                continue
            if "temporarily reduced" in err.lower() or "high demand" in err.lower():
                print(f"  ⏳ Cerebras high demand → wait 30s")
                time.sleep(30)
                continue
            return f"FAIL: {err[:200]}"
    return "FAIL: Cerebras retries exhausted"


def call_groq(model_id: str, prompt: str) -> str:
    """Call Groq API — free, fast."""
    try:
        from groq import Groq
    except ImportError:
        return "FAIL: pip install groq"

    if not GROQ_API_KEY:
        return "FAIL: GROQ_API_KEY not set in .env"

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=512,
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
                wait = min(2 ** attempt * 10, 120)
                print(f"  ⏳ Groq rate limit → wait {wait}s")
                time.sleep(wait)
                continue
            return f"FAIL: {err[:200]}"
    return "FAIL: Groq retries exhausted"

def call_openrouter(model_id: str, prompt: str) -> str:
    import requests
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return "FAIL: OPENROUTER_API_KEY not set"
    for attempt in range(6):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json={"model": model_id, "max_tokens": 512, "temperature": 0.0,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": prompt}]},
                timeout=60,
            )
            data = resp.json()
            if "error" in data:
                return f"FAIL: {str(data['error'])[:200]}"
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            wait = min(2 ** attempt * 10, 120)
            time.sleep(wait)
    return "FAIL: OpenRouter retries exhausted"

# ── CSV helpers ────────────────────────────────────────────────────────────────

def write_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  💾 Saved {len(rows)} rows → {path.name}")


# ── Core evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(model_key: str, data: list):
    cfg      = MODELS[model_key]
    label    = cfg["label"]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    out_path = RESULTS_DIR / cfg["out_csv"]

    print(f"\n{'━'*68}")
    print(f"  Model    : {label}")
    print(f"  Provider : {provider} / {model_id}")
    print(f"  Output   : {cfg['out_csv']}")
    print(f"{'━'*68}")

    # Check keys
    if provider == "cerebras" and not CEREBRAS_API_KEY:
        print("  ✗  CEREBRAS_API_KEY not set in .env")
        print("     Get free key at: https://cloud.cerebras.ai")
        return None
    elif provider == "openrouter" and not OPENROUTER_API_KEY:
        print("  ✗  OPENROUTER_API_KEY not set in .env")
        return None
    if provider == "groq" and not GROQ_API_KEY:
        print("  ✗  GROQ_API_KEY not set in .env")
        return None

    # Load checkpoint
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    done[row["id"]] = row

    remaining = len(data) - len(done)
    print(f"  Done     : {len(done)}/{len(data)}  |  Remaining: {remaining}")

    if remaining == 0:
        print("  ✓  Already complete — skipping.\n")
        return _compute_accuracy(done)

    # Delays — Cerebras is very fast, Groq is fast
    delay = 1.0 if provider == "cerebras" else 2.0

    results = list(done.values())

    for i, item in enumerate(data, 1):
        item_id = item["id"]
        task    = item["task_type"]

        if item_id in done:
            continue

        prompt = build_prompt(item)

        try:
            if provider == "cerebras":
                pred = call_cerebras(model_id, prompt)
            elif provider == "groq":
                pred = call_groq(model_id, prompt)
            elif provider == "openrouter":
                pred = call_openrouter(model_id, prompt)
            else:
                pred = f"FAIL: unknown provider {provider}"
        except Exception as e:
            pred = f"FAIL: {str(e)[:200]}"

        correct = score_answer(item["answer"], pred, task)
        status  = "✓" if correct else "✗"

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

        print(f"  [{i:03d}/{len(data)}] {status}  {item_id:<14}  {pred[:60]}")

        if len(results) % 20 == 0:
            write_csv(results, out_path)

        time.sleep(delay)

    write_csv(results, out_path)
    acc = _compute_accuracy(done)
    _print_summary(acc, label)
    return acc


# ── Accuracy helpers ───────────────────────────────────────────────────────────

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
    print(f"\n  ── {label} Final Results ──")
    for abbr in ["REG", "NUM", "CON", "TMP"]:
        print(f"    {abbr}: {acc.get(abbr, 0)*100:.1f}%")
    print(f"    Overall: {acc.get('overall', 0)*100:.1f}%  (n={acc.get('n', 0)})")


def print_leaderboard(new_results: dict):
    existing = [
        {"label": "Claude 3 Haiku",    "REG": 92.5, "NUM": 93.8, "CON": 86.7, "TMP": 91.4, "Overall": 91.3, "note": "(150-item)"},
        {"label": "Gemini 2.5 Flash",  "REG": 95.5, "NUM": 80.9, "CON": 88.6, "TMP": 87.0, "Overall": 89.7, "note": "(406-item, partial)"},
        {"label": "Qwen3-32B",         "REG": 85.1, "NUM": 77.2, "CON": 90.3, "TMP": 92.3, "Overall": 85.5, "note": ""},
        {"label": "LLaMA-3.3-70B",     "REG": 86.2, "NUM": 75.0, "CON": 95.2, "TMP": 79.5, "Overall": 83.7, "note": ""},
        {"label": "Llama 4 Scout 17B", "REG": 86.2, "NUM": 66.3, "CON": 98.4, "TMP": 84.6, "Overall": 83.3, "note": ""},
        {"label": "LLaMA-3-8B",        "REG": 79.9, "NUM": 64.1, "CON": 93.5, "TMP": 78.2, "Overall": 78.1, "note": ""},
        {"label": "Mistral-7B",        "REG": 79.9, "NUM": 66.3, "CON": 80.6, "TMP": 74.4, "Overall": 75.9, "note": ""},
        {"label": "DeepSeek R1 70B",   "REG": 72.4, "NUM": 69.6, "CON": 96.8, "TMP": 70.5, "Overall": 75.1, "note": ""},
        {"label": "Gemma 4 E4B",       "REG": 83.9, "NUM": 50.0, "CON": 72.6, "TMP": 62.8, "Overall": 70.4, "note": ""},
    ]

    for key, acc in new_results.items():
        existing.append({
            "label":   MODELS[key]["label"],
            "REG":     round(acc.get("REG", 0) * 100, 1),
            "NUM":     round(acc.get("NUM", 0) * 100, 1),
            "CON":     round(acc.get("CON", 0) * 100, 1),
            "TMP":     round(acc.get("TMP", 0) * 100, 1),
            "Overall": round(acc.get("overall", 0) * 100, 1),
            "note":    "",
        })

    existing.sort(key=lambda x: x["Overall"], reverse=True)

    print(f"\n{'━'*90}")
    print(f"  INDIAFINBENCH — FULL LEADERBOARD ({len(existing)} models, 406 items)")
    print(f"{'━'*90}")
    print(f"  {'#':<3} {'Model':<26} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6} {'Overall':>8}")
    print(f"  {'─'*3} {'─'*26} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")

    for rank, m in enumerate(existing, 1):
        note = m.get("note", "")
        print(
            f"  {rank:<3} {m['label']:<26} "
            f"{m['REG']:>5.1f}% {m['NUM']:>5.1f}% "
            f"{m['CON']:>5.1f}% {m['TMP']:>5.1f}% "
            f"{m['Overall']:>7.1f}% {note}"
        )

    print(f"\n  Human Expert Baseline (n=30): 60.0%")
    print(f"{'━'*90}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate new models on IndiaFinBench"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to evaluate"
    )
    args = parser.parse_args()

    print(f"\n{'━'*68}")
    print(f"  IndiaFinBench — New Model Evaluations (v2)")
    print(f"  Models   : {', '.join(args.models)}")
    print(f"  Cerebras : {'✓ key set' if CEREBRAS_API_KEY else '✗ CEREBRAS_API_KEY missing'}")
    print(f"  Groq     : {'✓ key set' if GROQ_API_KEY else '✗ GROQ_API_KEY missing'}")
    print(f"{'━'*68}")

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n  Loaded {len(data)} items from {QA_PATH.name}\n")

    new_results = {}
    for key in args.models:
        acc = evaluate_model(key, data)
        if acc is not None:
            new_results[key] = acc

    if new_results:
        print_leaderboard(new_results)

    print("\n  ✓  Done. Upload the result CSVs to get the final leaderboard.\n")


if __name__ == "__main__":
    main()
