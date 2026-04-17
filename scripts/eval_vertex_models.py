"""
eval_vertex_models.py
---------------------
Evaluate Claude 3.5 Sonnet and Gemini 2.5 Pro on full 406-item benchmark
via Google Cloud Vertex AI (billed against GCP free credits).

GCP Project: finindiabench
Required env vars (set before running):
  export GOOGLE_CLOUD_PROJECT=finindiabench
  export GOOGLE_CLOUD_LOCATION=us-central1
  export GOOGLE_APPLICATION_CREDENTIALS="D:/Projects/IndiaFinBench/gcp-key.json"

Usage:
  python scripts/eval_vertex_models.py --model claude        # Claude 3.5 Sonnet
  python scripts/eval_vertex_models.py --model gemini        # Gemini 2.5 Pro
  python scripts/eval_vertex_models.py --model all           # both

Resumes from partial CSV on restart — safe to interrupt.
"""

import argparse, io, json, os, re, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.novel_methods_utils import load_dataset

try:
    from rapidfuzz import fuzz as _rf
except ImportError:
    raise ImportError("Run: pip install rapidfuzz")

BASE        = Path(__file__).parent.parent
RESULTS_DIR = BASE / "evaluation/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Vertex AI model IDs ──────────────────────────────────────────────────────
# If you get a "model not found" error, go to:
#   https://console.cloud.google.com/vertex-ai/model-garden?project=finindiabench
# and search for the model to find its exact publisher/model ID.

MODELS = {
    "claude": {
        "label":   "Claude 3.5 Sonnet",
        "model_id": "claude-3-5-sonnet-v2@20241022",   # Vertex Model Garden ID
        "out_csv": "claude35_sonnet_results.csv",
    },
    "gemini": {
        "label":   "Gemini 2.5 Pro",
        "model_id": "gemini-2.5-pro-preview-05-06",    # Vertex API ID
        "out_csv": "gemini25_pro_results.csv",
    },
}

SYSTEM_PROMPT = (
    "You are an expert in Indian financial regulation and policy. "
    "Answer questions using ONLY the provided context passage. "
    "Do not use any external knowledge. Be concise and precise. "
    "Give only the answer — no preamble."
)


# ── Scoring (identical to existing evaluate.py pipeline) ─────────────────────
def score_prediction(prediction: str, reference: str, task_type: str) -> int:
    pred = str(prediction).lower().strip()
    ref  = str(reference).lower().strip()

    # 1. Exact match
    if pred == ref:
        return 1

    # 2. Fuzzy token match (threshold=72, same as main eval)
    if _rf.token_set_ratio(pred, ref) >= 72:
        return 1

    # 3. Numerical extraction match
    def extract_nums(s):
        s = re.sub(r'[₹,]', '', s)
        return set(re.findall(r'\d+(?:\.\d+)?', s))
    pn, rn = extract_nums(pred), extract_nums(ref)
    if pn and rn and pn == rn:
        return 1

    # 4. Yes/No for contradiction detection
    if 'contradiction' in str(task_type).lower():
        p0 = (pred.split() or [''])[0]
        r0 = (ref.split()  or [''])[0]
        return 1 if p0 == r0 else 0

    return 0


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(row) -> str:
    task    = str(row.get('task_type', '')).lower()
    context = str(row.get('context', ''))
    question = str(row.get('question', ''))

    if 'contradiction' in task:
        ctx_a = str(row.get('context_a', context))[:1500]
        ctx_b = str(row.get('context_b', ''))[:800]
        ctx_block = f"PASSAGE A:\n{ctx_a}\n\nPASSAGE B:\n{ctx_b}"
        instruction = ("Answer Yes or No, then give a one-sentence explanation "
                       "of why the passages contradict or don't contradict.")
    elif 'numerical' in task:
        ctx_block   = f"CONTEXT:\n{context}"
        instruction = "Show your calculation steps and include the unit in your final answer."
    else:
        ctx_block   = f"CONTEXT:\n{context}"
        instruction = "Give the exact answer as it appears in or is derivable from the context."

    return (f"{SYSTEM_PROMPT}\n\n"
            f"{ctx_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"{instruction}")


# ── Vertex AI client ──────────────────────────────────────────────────────────
def get_vertex_client():
    project  = os.environ.get("GOOGLE_CLOUD_PROJECT", "finindiabench")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    try:
        from google import genai
        from google.genai.types import HttpOptions
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        print(f"Vertex AI client ready  project={project}  location={location}")
        return client
    except Exception as e:
        print(f"ERROR: Could not create Vertex AI client: {e}")
        print("Make sure GOOGLE_APPLICATION_CREDENTIALS is exported (not set).")
        sys.exit(1)


def call_vertex(client, model_id: str, prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            time.sleep(0.3)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                # Model ID is wrong — give a helpful message and stop
                print(f"\nERROR: Model '{model_id}' not found on Vertex AI.")
                print("To find the correct model ID:")
                print("  1. Go to https://console.cloud.google.com/vertex-ai/model-garden?project=finindiabench")
                print("  2. Search for 'Claude' or 'Gemini'")
                print("  3. Click the model card and copy the model ID shown")
                print("  4. Edit MODELS dict in this script and re-run")
                sys.exit(1)
            elif "429" in err or "quota" in err.lower() or "exhausted" in err.lower():
                wait = 60 * (attempt + 1)
                print(f"  Rate limit (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"  Server overload (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Vertex error attempt {attempt+1}: {e}")
                time.sleep(5)
    print("  Max retries reached — returning empty string for this item")
    return ""


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate_model(key: str, dataset, client):
    cfg      = MODELS[key]
    label    = cfg["label"]
    model_id = cfg["model_id"]
    out_path = RESULTS_DIR / cfg["out_csv"]

    # Resume from partial
    if out_path.exists():
        existing  = __import__('pandas').read_csv(out_path)
        done_ids  = set(existing['id'].astype(str).tolist())
        results   = existing.to_dict('records')
        remaining = len(dataset) - len(done_ids)
        print(f"  Resuming {label}: {len(done_ids)} done, {remaining} remaining")
    else:
        done_ids, results = set(), []

    total = len(dataset)
    for i, (_, row) in enumerate(dataset.iterrows()):
        item_id = str(row['id'])
        if item_id in done_ids:
            continue

        prompt     = build_prompt(row)
        reference  = str(row.get('answer', ''))
        prediction = call_vertex(client, model_id, prompt)

        if prediction == "" and i < 5:
            print(f"    WARNING: empty response for item {item_id}")

        correct = score_prediction(prediction, reference,
                                   str(row.get('task_type', '')))
        results.append({
            'id':            item_id,
            'task_type':     row.get('task_type', ''),
            'difficulty':    row.get('difficulty', ''),
            'question':      str(row.get('question', ''))[:200],
            'ref_answer':    reference,
            'prediction':    prediction[:500],
            'correct':       correct,
            'model_version': model_id,
        })
        done_ids.add(item_id)

        # Checkpoint every 25 items
        if len(results) % 25 == 0:
            import pandas as pd
            pd.DataFrame(results).to_csv(out_path, index=False)
            pct = len(results) / total * 100
            acc = sum(r['correct'] for r in results) / len(results) * 100
            print(f"  [{label}] {len(results)}/{total} ({pct:.0f}%)  running acc={acc:.1f}%")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    acc = df['correct'].mean() * 100
    print(f"\n  {label} COMPLETE — {acc:.1f}% accuracy  ({len(df)} items)")
    print(f"  Saved → {out_path}")

    # Per-task breakdown
    for task, grp in df.groupby('task_type'):
        short = {'regulatory_interpretation':'REG','numerical_reasoning':'NUM',
                 'contradiction_detection':'CON','temporal_reasoning':'TMP'}.get(task, task[:3].upper())
        print(f"    {short}: {grp['correct'].mean()*100:.1f}% (n={len(grp)})")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Claude 3.5 Sonnet and/or Gemini 2.5 Pro via Vertex AI")
    parser.add_argument(
        "--model",
        choices=["claude", "gemini", "all"],
        default="all",
        help="Which model to evaluate (default: all)")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"Dataset loaded: {len(dataset)} items\n")

    client = get_vertex_client()

    keys_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for key in keys_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {MODELS[key]['label']}")
        print(f"Model ID:   {MODELS[key]['model_id']}")
        print(f"Output:     {MODELS[key]['out_csv']}")
        print('='*60)
        evaluate_model(key, dataset, client)

    print("\nAll evaluations complete.")
