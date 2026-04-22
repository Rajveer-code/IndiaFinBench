"""
eval_vertex_models.py
---------------------
Evaluate Claude Sonnet 4.6 and Gemini 2.5 Pro on full 406-item benchmark
via Google Cloud Vertex AI (billed against GCP free credits).

IMPORTANT SDK split:
  Claude  → anthropic.AnthropicVertex  (NOT google-genai)
  Gemini  → google-genai               (as before)

GCP Project: finindiabench
Required env vars (set before running):
  export GOOGLE_CLOUD_PROJECT=finindiabench
  export GOOGLE_APPLICATION_CREDENTIALS="D:/Projects/IndiaFinBench/gcp-key.json"

Usage:
  python scripts/eval_vertex_models.py --model claude        # Claude Sonnet 4.6
  python scripts/eval_vertex_models.py --model gemini        # Gemini 2.5 Pro
  python scripts/eval_vertex_models.py --model all           # both

Resumes from partial CSV on restart — safe to interrupt.
"""

import argparse, io, os, re, sys, time
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

GCP_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "finindiabench")
GEMINI_LOC  = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
CLAUDE_LOC  = "us-east5"   # Claude on Vertex AI is only available in us-east5

MODELS = {
    "claude": {
        "label":    "Claude Sonnet 4.6",
        "model_id": "claude-sonnet-4-6",        # Anthropic SDK model ID on Vertex
        "out_csv":  "claude_sonnet46_results.csv",
        "backend":  "anthropic",
    },
    "gemini": {
        "label":    "Gemini 2.5 Pro",
        "model_id": "gemini-2.5-pro",
        "out_csv":  "gemini25_pro_results.csv",
        "backend":  "google",
    },
}

SYSTEM_PROMPT = (
    "You are an expert in Indian financial regulation and policy. "
    "Answer questions using ONLY the provided context passage. "
    "Do not use any external knowledge. Be concise and precise. "
    "Give only the answer — no preamble."
)


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_prediction(prediction: str, reference: str, task_type: str) -> int:
    pred = str(prediction).lower().strip()
    ref  = str(reference).lower().strip()

    if pred == ref:
        return 1
    if _rf.token_set_ratio(pred, ref) >= 72:
        return 1

    def extract_nums(s):
        s = re.sub(r'[₹,]', '', s)
        return set(re.findall(r'\d+(?:\.\d+)?', s))
    pn, rn = extract_nums(pred), extract_nums(ref)
    if pn and rn and pn == rn:
        return 1

    # YES/NO for contradiction — use regex to handle "Yes, ..." "No. ..." etc.
    if 'contradiction' in str(task_type).lower():
        p0 = re.search(r'\b(yes|no)\b', pred)
        r0 = re.search(r'\b(yes|no)\b', ref)
        if p0 and r0:
            return 1 if p0.group() == r0.group() else 0
        return 0

    return 0


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(row) -> str:
    """User-facing part of the prompt — system prefix added by each call function."""
    task     = str(row.get('task_type', '')).lower()
    context  = str(row.get('context', ''))
    question = str(row.get('question', ''))

    if 'contradiction' in task:
        ctx_a = str(row.get('context_a', context))[:1500]
        ctx_b = str(row.get('context_b', ''))[:800]
        ctx_block   = f"PASSAGE A:\n{ctx_a}\n\nPASSAGE B:\n{ctx_b}"
        instruction = ("Answer Yes or No, then give a one-sentence explanation "
                       "of why the passages contradict or don't contradict.")
    elif 'numerical' in task:
        ctx_block   = f"CONTEXT:\n{context}"
        instruction = "Show your calculation steps and include the unit in your final answer."
    else:
        ctx_block   = f"CONTEXT:\n{context}"
        instruction = "Give the exact answer as it appears in or is derivable from the context."

    return f"{ctx_block}\n\nQUESTION: {question}\n\n{instruction}"


# ── Anthropic SDK client  (Claude on Vertex AI) ───────────────────────────────
def get_claude_client():
    try:
        from anthropic import AnthropicVertex
    except ImportError:
        print("ERROR: anthropic package not installed.")
        print("Run:  pip install anthropic")
        sys.exit(1)
    try:
        client = AnthropicVertex(project_id=GCP_PROJECT, region=CLAUDE_LOC)
        print(f"AnthropicVertex client ready  project={GCP_PROJECT}  region={CLAUDE_LOC}")
        return client
    except Exception as e:
        print(f"ERROR creating AnthropicVertex client: {e}")
        sys.exit(1)


def call_claude(client, model_id: str, prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            time.sleep(0.4)
            return msg.content[0].text.strip()
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                print(f"\nERROR: Claude model '{model_id}' not found on Vertex AI (region={CLAUDE_LOC}).")
                print("Common model IDs to try — update MODELS['claude']['model_id']:")
                print("  claude-3-5-sonnet-v2@20241022  (Claude 3.5 Sonnet — definitely available)")
                print("  claude-3-7-sonnet@20250219     (Claude 3.7 Sonnet)")
                print("  claude-sonnet-4-5@20250514     (Claude Sonnet 4 — if enabled)")
                print("  claude-sonnet-4-6              (Claude Sonnet 4.6 — if enabled)")
                print("Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude")
                sys.exit(1)
            elif "403" in err or "permission" in err.lower() or "tos" in err.lower() or "terms" in err.lower():
                print(f"\nERROR: Permission denied for Claude on Vertex AI.")
                print("Fix: Visit Model Garden in GCP Console and click Enable for Claude:")
                print("  https://console.cloud.google.com/vertex-ai/model-garden?project=finindiabench")
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
                print(f"  Claude error attempt {attempt+1}: {e}")
                time.sleep(5)
    print("  Max retries reached — returning empty string for this item")
    return ""


# ── Google genai client  (Gemini on Vertex AI) ────────────────────────────────
def get_gemini_client():
    try:
        from google import genai
        from google.genai.types import HttpOptions
        client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GEMINI_LOC,
            http_options=HttpOptions(api_version="v1"),
        )
        print(f"Gemini Vertex client ready  project={GCP_PROJECT}  location={GEMINI_LOC}")
        return client
    except Exception as e:
        print(f"ERROR: Could not create Gemini Vertex client: {e}")
        sys.exit(1)


def call_gemini(client, model_id: str, prompt: str, max_retries: int = 5) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=full_prompt,
            )
            time.sleep(0.3)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower():
                print(f"\nERROR: Gemini model '{model_id}' not found on Vertex AI.")
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
                print(f"  Gemini error attempt {attempt+1}: {e}")
                time.sleep(5)
    print("  Max retries reached — returning empty string for this item")
    return ""


# ── Main evaluation loop ──────────────────────────────────────────────────────
def evaluate_model(key: str, dataset):
    cfg      = MODELS[key]
    label    = cfg["label"]
    model_id = cfg["model_id"]
    backend  = cfg["backend"]
    out_path = RESULTS_DIR / cfg["out_csv"]

    if backend == "anthropic":
        client  = get_claude_client()
        call_fn = call_claude
    else:
        client  = get_gemini_client()
        call_fn = call_gemini

    # Resume from partial CSV
    if out_path.exists():
        import pandas as pd
        existing  = pd.read_csv(out_path)
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

        prompt     = build_user_prompt(row)
        reference  = str(row.get('answer', ''))
        prediction = call_fn(client, model_id, prompt)

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
    print(f"\n  {label} COMPLETE -- {acc:.1f}% accuracy  ({len(df)} items)")
    print(f"  Saved -> {out_path}")

    for task, grp in df.groupby('task_type'):
        short = {'regulatory_interpretation':'REG','numerical_reasoning':'NUM',
                 'contradiction_detection':'CON','temporal_reasoning':'TMP'}.get(task, task[:3].upper())
        print(f"    {short}: {grp['correct'].mean()*100:.1f}% (n={len(grp)})")

    return df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Claude Sonnet 4.6 and/or Gemini 2.5 Pro via Vertex AI")
    parser.add_argument(
        "--model", choices=["claude", "gemini", "all"], default="all",
        help="Which model to evaluate (default: all)")
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"Dataset loaded: {len(dataset)} items\n")

    keys_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for key in keys_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {MODELS[key]['label']}")
        print(f"Model ID:   {MODELS[key]['model_id']}")
        print(f"Output:     {MODELS[key]['out_csv']}")
        print('='*60)
        evaluate_model(key, dataset)

    print("\nAll evaluations complete.")
