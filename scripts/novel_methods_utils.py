"""Shared utilities for all novel method scripts.

Verified field names (from Section 16 Step 3 & 4):
  Dataset: id, task_type, source, document, regulation, context, question,
           answer, answer_type, difficulty, context_a, context_b, explanation
  Results: id, task_type, difficulty, question, ref_answer, prediction,
           correct, model_version
"""
import json, os, pandas as pd, numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATASET_PATH = REPO_ROOT / "annotation/raw_qa/indiafinbench_qa_combined_406.json"
RESULTS_DIR = REPO_ROOT / "evaluation/results"
OUTPUT_DIR = REPO_ROOT / "evaluation/novel_methods"
FIGURES_DIR = REPO_ROOT / "paper/figures/novel_methods"
TABLES_DIR = REPO_ROOT / "paper/tables/novel_methods"

TASK_MAP = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning": "NUM",
    "contradiction_detection": "CON",
    "temporal_reasoning": "TMP"
}

MODEL_FILES = {
    "Gemini 2.5 Flash":    "gemini_results.csv",
    "Qwen3-32B":           "qwen3_32b_results.csv",
    "LLaMA-3.3-70B":       "groq70b_results.csv",
    "Llama 4 Scout 17B":   "llama4scout_results.csv",
    "Kimi K2":             "kimi_k2_results.csv",
    "LLaMA-3-8B":          "llama3_results.csv",
    "GPT-OSS 120B":        "gpt_oss_120b_results.csv",
    "GPT-OSS 20B":         "gpt_oss_20b_results.csv",
    "Mistral-7B":          "mistral_results.csv",
    "DeepSeek R1 70B":     "deepseek_r1_70b_results.csv",
    "Gemma 4 E4B":         "gemma4_e4b_results.csv",
}


def load_dataset():
    """Load dataset JSON. Returns DataFrame with verified field names."""
    with open(DATASET_PATH, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('items', data.get('data', list(data.values())[0]))
    return pd.DataFrame(data)


def load_all_results():
    """Returns dict: model_name -> DataFrame.

    Verified columns in each CSV:
      id, task_type, difficulty, question, ref_answer, prediction,
      correct, model_version
    """
    results = {}
    for model_name, fname in MODEL_FILES.items():
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df['model'] = model_name
            results[model_name] = df
        else:
            print(f"WARNING: {fpath} not found, skipping {model_name}")
    return results


def _correctness_col(df):
    """Return the name of the correctness column (0/1)."""
    # Exact match first, then substring
    for candidate in ['correct', 'is_correct', 'Correct']:
        if candidate in df.columns:
            return candidate
    matches = [c for c in df.columns if 'correct' in c.lower()]
    if matches:
        return matches[0]
    raise KeyError(f"No correctness column found. Available: {df.columns.tolist()}")


def _id_col(df):
    """Return the item-ID column name."""
    for candidate in ['id', 'item_id', 'ID']:
        if candidate in df.columns:
            return candidate
    matches = [c for c in df.columns if c.lower() in ('id', 'item_id', 'idx')]
    return matches[0] if matches else df.columns[0]


def _task_col(df):
    """Return the task-type column name."""
    for candidate in ['task_type', 'task']:
        if candidate in df.columns:
            return candidate
    matches = [c for c in df.columns if 'task' in c.lower()]
    return matches[0] if matches else None


def _prediction_col(df):
    """Return the prediction/output column name."""
    for candidate in ['prediction', 'model_prediction', 'output']:
        if candidate in df.columns:
            return candidate
    matches = [c for c in df.columns if 'predict' in c.lower() or 'output' in c.lower()]
    return matches[0] if matches else None


def _reference_col(df):
    """Return the reference/gold-answer column name.

    Handles both 'ref_answer' (results CSV) and 'reference_answer' (legacy).
    Also handles 'answer' (dataset).
    """
    for candidate in ['ref_answer', 'reference_answer', 'gold_answer', 'answer']:
        if candidate in df.columns:
            return candidate
    matches = [c for c in df.columns
               if 'ref' in c.lower() or 'gold' in c.lower() or 'answer' in c.lower()]
    return matches[0] if matches else None


def load_correctness_matrix():
    """Returns DataFrame: rows=items (by id), cols=models, values=0/1."""
    results = load_all_results()
    matrices = []
    for model_name, df in results.items():
        corr = _correctness_col(df)
        id_c = _id_col(df)
        tmp = df[[id_c, corr]].copy()
        tmp.columns = ['item_id', model_name]
        matrices.append(tmp.set_index('item_id'))
    return pd.concat(matrices, axis=1)


def get_task_accuracies():
    """Returns DataFrame: rows=models, cols=[REG,NUM,CON,TMP,Overall]."""
    results = load_all_results()
    rows = []
    for model_name, df in results.items():
        corr = _correctness_col(df)
        task_c = _task_col(df)
        row = {'Model': model_name, 'Overall': df[corr].mean()}
        if task_c:
            for task_full, task_short in TASK_MAP.items():
                mask = df[task_c].str.contains(task_full, case=False, na=False)
                row[task_short] = df.loc[mask, corr].mean() if mask.sum() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


# ── API helpers ────────────────────────────────────────────────────────────────
import time

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")


def call_gemini(prompt: str, max_retries: int = 3, sleep: int = 4) -> str:
    """Call Gemini 2.5 Flash with rate limiting (15 req/min free tier)."""
    if not GEMINI_KEY:
        raise ValueError("Set GEMINI_API_KEY environment variable")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-latest")
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(sleep)
            return response.text
        except Exception as e:
            print(f"Gemini error (attempt {attempt+1}): {e}")
            time.sleep(10)
    return ""


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile",
              max_retries: int = 3) -> str:
    """Call Groq API."""
    if not GROQ_KEY:
        raise ValueError("Set GROQ_API_KEY environment variable")
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0
            )
            time.sleep(2)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return ""
