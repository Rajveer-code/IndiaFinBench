"""
evaluate_new_models.py
Evaluates new models on IndiaFinBench (150 items).

Models:
  llama4scout  — Llama 4 Scout 17B  (Groq)
  qwen3_32b    — Qwen3-32B           (Groq)
  deepseek_r1  — DeepSeek-R1         (OpenRouter, strips <think> blocks)
  gemma3_12b   — Gemma 3 12B         (Ollama, skipped if not pulled)
  phi4         — Phi-4               (Ollama, skipped if not pulled)

Usage:
  python scripts/evaluate_new_models.py
  python scripts/evaluate_new_models.py --models llama4scout qwen3_32b
"""
import json, csv, os, re, time, sys, io, argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("pip install rapidfuzz")

BASE = Path(__file__).parent.parent
QA_PATH = BASE / "annotation/raw_qa/indiafinbench_qa_combined_150.json"
RESULTS_DIR = BASE / "evaluation/results"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = {
    "llama4scout": {
        "label": "Llama 4 Scout 17B",
        "provider": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "qwen3_32b": {
        "label": "Qwen3-32B",
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "extra_params": {"reasoning_effort": "none"},  # disable thinking mode
    },
    "deepseek_r1": {
        "label": "DeepSeek-R1",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1",
        "strip_think": True,
    },
    "gemma3_12b": {
        "label": "Gemma 3 12B",
        "provider": "ollama",
        "model_id": "gemma3:12b",
    },
    "phi4": {
        "label": "Phi-4",
        "provider": "ollama",
        "model_id": "phi4",
    },
}

SYSTEM_PROMPT = """You are an expert in Indian financial regulation and policy.
Answer questions using ONLY the provided context passage.
Do not use any external knowledge.
Be concise and precise. Give only the answer -- no preamble."""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q = item["question"]
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


def call_groq(model_id: str, prompt: str, extra_params: dict = None) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    params = {
        "model": model_id,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if extra_params:
        params.update(extra_params)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(2 ** attempt * 5, 60)
                print(f"  Groq rate limit -> wait {wait}s")
                time.sleep(wait)
                continue
            return f"FAIL: {err[:150]}"
    return "FAIL: Groq retries exhausted"


def call_openrouter(model_id: str, prompt: str) -> str:
    import requests
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/IndiaFinBench",
        "X-Title": "IndiaFinBench",
    }
    payload = {
        "model": model_id,
        "max_tokens": 2048,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    for attempt in range(5):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 10, 120)
                print(f"  OpenRouter rate limit -> wait {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"FAIL: {str(e)[:150]}"
    return "FAIL: OpenRouter retries exhausted"


def call_ollama(model_id: str, prompt: str) -> str:
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_id,
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 300},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def check_ollama_model(model_id: str) -> bool:
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(model_id.split(":")[0] in m for m in models)
    except Exception:
        return False


def evaluate_model(model_key: str, data: list):
    cfg = MODELS[model_key]
    label = cfg["label"]
    provider = cfg["provider"]
    model_id = cfg["model_id"]
    should_strip_think = cfg.get("strip_think", False)
    extra_params = cfg.get("extra_params", {})
    out_path = RESULTS_DIR / f"{model_key}_results.csv"

    # Check availability
    if provider == "ollama":
        if not check_ollama_model(model_id):
            print(f"\nSkipping {label}: not in Ollama (run: ollama pull {model_id})")
            return
    elif provider == "openrouter" and not OPENROUTER_API_KEY:
        print(f"\nSkipping {label}: OPENROUTER_API_KEY not set")
        return
    elif provider == "groq" and not GROQ_API_KEY:
        print(f"\nSkipping {label}: GROQ_API_KEY not set")
        return

    # Load checkpoint
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "FAIL" not in row.get("prediction", ""):
                    done[row["id"]] = row

    remaining = len(data) - len(done)
    print(f"\n{'='*65}")
    print(f"  Model    : {label}")
    print(f"  Provider : {provider} / {model_id}")
    print(f"  Done     : {len(done)}/150  |  Remaining: {remaining}")
    print(f"{'='*65}")

    if remaining == 0:
        print("  Already complete -- skipping.")
        _print_summary(done, label)
        return

    results = list(done.values())
    delays = {"ollama": 0.2, "groq": 0.5, "openrouter": 1.0}
    delay = delays.get(provider, 1.0)

    for i, item in enumerate(data):
        item_id = item["id"]
        if item_id in done:
            continue

        prompt = build_prompt(item)
        try:
            if provider == "groq":
                pred = call_groq(model_id, prompt, extra_params if extra_params else None)
            elif provider == "openrouter":
                pred = call_openrouter(model_id, prompt)
            elif provider == "ollama":
                pred = call_ollama(model_id, prompt)
            else:
                pred = f"FAIL: unknown provider {provider}"
        except Exception as e:
            pred = f"FAIL: {str(e)[:150]}"

        if should_strip_think:
            pred = strip_think(pred)

        correct = score_answer(item["answer"], pred, item["task_type"])

        row = {
            "id": item_id,
            "task_type": item["task_type"],
            "difficulty": item.get("difficulty", ""),
            "question": item["question"][:80],
            "ref_answer": item["answer"],
            "prediction": pred[:300],
            "correct": correct,
        }
        results.append(row)

        status = "OK" if correct else "--"
        print(f"  [{len(results):3d}/150] {status} {item_id:<12} {pred[:55]}")

        # Write checkpoint every 10
        if len(results) % 10 == 0:
            _write_csv(results, out_path)

        if delay:
            time.sleep(delay)

    _write_csv(results, out_path)
    _print_summary({r["id"]: r for r in results}, label)


def _write_csv(results: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id","task_type","difficulty","question","ref_answer","prediction","correct"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"  [checkpoint] Saved {len(results)} rows -> {path.name}")


def _print_summary(done_dict: dict, label: str):
    from collections import defaultdict
    task_scores = defaultdict(list)
    task_abbr = {
        "regulatory_interpretation": "REG",
        "numerical_reasoning": "NUM",
        "contradiction_detection": "CON",
        "temporal_reasoning": "TMP",
    }
    for row in done_dict.values():
        task_scores[row["task_type"]].append(int(row["correct"]))

    print(f"\n  --- {label} Results ---")
    overall = []
    for task, abbr in task_abbr.items():
        scores = task_scores.get(task, [])
        if scores:
            acc = sum(scores) / len(scores) * 100
            print(f"    {abbr}: {acc:.1f}% ({sum(scores)}/{len(scores)})")
            overall.extend(scores)
    if overall:
        print(f"    Overall: {sum(overall)/len(overall)*100:.1f}% ({sum(overall)}/{len(overall)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Models to evaluate")
    args = parser.parse_args()

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {QA_PATH.name}")

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
            continue
        evaluate_model(model_key, data)

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
