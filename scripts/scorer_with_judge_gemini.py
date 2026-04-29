"""
scorer_with_judge_gemini.py
============================
IndiaFinBench Hybrid Scoring Pipeline — using Gemini 2.5 Flash as judge.
Uses Google Cloud Vertex AI with your GCP credits (project: finindiabench).

SETUP (one time):
    gcloud auth application-default login
    gcloud config set project finindiabench

OR set environment variable:
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT="finindiabench"

USAGE:
    python scorer_with_judge_gemini.py \
        --dataset indiafinbench_qa_combined_406.json \
        --results_dir evaluation/results \
        --output evaluation/results_judged

WHAT IT DOES:
    1. Loads each model's result CSV
    2. For items scored INCORRECT on NUM, REG, TMP tasks:
       - Calls Gemini 2.5 Flash as semantic judge
       - If judge says CORRECT, flips the score to 1
    3. Saves corrected CSVs + full audit log + comparison table

COST ESTIMATE:
    ~400-600 incorrect items across 12 models × ~200 tokens per call
    = ~100,000 tokens total = approximately Rs. 8-15 from your GCP credits
"""

import csv
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

# ── Judge prompt ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator for an Indian financial regulatory reasoning benchmark.

Question: {question}

Reference Answer: {reference}

Model Prediction: {prediction}

Determine if the model prediction is CORRECT.

Rules:
- CORRECT if the final value or fact matches, even if:
  * Different formatting (₹ vs Rs. vs INR, commas in numbers, units)
  * Extra calculation steps shown before the final answer
  * Paraphrasing of the same regulatory rule
  * Rounding within 1%
- INCORRECT only if: wrong number, wrong threshold, wrong date, wrong entity, contradicts reference

Reply with exactly:
CORRECT
Reason: [one sentence]

OR

INCORRECT  
Reason: [one sentence]"""

# ── Task configuration ────────────────────────────────────────────────────────

JUDGE_TASKS = {"numerical_reasoning", "regulatory_interpretation", "temporal_reasoning"}

TASK_ABBR = {
    "regulatory_interpretation": "REG",
    "numerical_reasoning":       "NUM",
    "contradiction_detection":   "CON",
    "temporal_reasoning":        "TMP",
}

# Model result file mapping
MODEL_FILES = {
    "gemini":           "gemini_results.csv",
    "qwen3_32b":        "qwen3_32b_results.csv",
    "groq70b":          "groq70b_results.csv",
    "llama4scout":      "llama4scout_results.csv",
    "kimi_k2":          "kimi_k2_results.csv",
    "llama3":           "llama3_results.csv",
    "gpt_oss_120b":     "gpt_oss_120b_results.csv",
    "gpt_oss_20b":      "gpt_oss_20b_results.csv",
    "mistral":          "mistral_results.csv",
    "deepseek_r1_70b":  "deepseek_r1_70b_results.csv",
    "gemma4_e4b":       "gemma4_e4b_results.csv",
    "gemini25_pro":     "gemini25_pro_results.csv",
}

MODEL_LABELS = {
    "gemini":           "Gemini 2.5 Flash",
    "qwen3_32b":        "Qwen3-32B",
    "groq70b":          "LLaMA-3.3-70B",
    "llama4scout":      "Llama 4 Scout 17B",
    "kimi_k2":          "Kimi K2",
    "llama3":           "LLaMA-3-8B",
    "gpt_oss_120b":     "GPT-OSS 120B",
    "gpt_oss_20b":      "GPT-OSS 20B",
    "mistral":          "Mistral-7B",
    "deepseek_r1_70b":  "DeepSeek R1 70B",
    "gemma4_e4b":       "Gemma 4 E4B",
    "gemini25_pro":     "Gemini 2.5 Pro",
}

# ── Gemini judge client ───────────────────────────────────────────────────────

def get_gemini_client():
    """Initialize Gemini client via Vertex AI or AI Studio."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "finindiabench")
    
    # Try Vertex AI first (uses GCP credits)
    try:
        from google import genai
        from google.genai.types import HttpOptions
        client = genai.Client(
            vertexai=True,
            project=project,
            location="us-central1",
            http_options=HttpOptions(api_version="v1"),
        )
        print(f"Using Vertex AI (project: {project}) — charges to GCP credits")
        return client, "vertex"
    except Exception as e:
        print(f"Vertex AI failed: {e}")
    
    # Fallback: AI Studio API key
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("Using AI Studio API key")
        return client, "aistudio"
    
    raise RuntimeError(
        "No Gemini credentials found.\n"
        "Option 1 (GCP credits): run 'gcloud auth application-default login'\n"
        "Option 2 (AI Studio): set GOOGLE_API_KEY environment variable"
    )


def call_judge(client, client_type: str, question: str,
               reference: str, prediction: str) -> dict:
    """Call Gemini judge. Returns {verdict: bool, reason: str}"""
    prompt = JUDGE_PROMPT.format(
        question=question[:600],
        reference=reference[:300],
        prediction=prediction[:500],
    )
    
    for attempt in range(5):
        try:
            if client_type == "vertex":
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
            else:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.0, "max_output_tokens": 150,
                            "thinking_config": {"thinking_budget": 0}},
                )
            
            raw = resp.text.strip() if resp.text else ""
            first_line = raw.split('\n')[0].strip().upper()
            verdict = first_line.startswith("CORRECT") and not first_line.startswith("INCORRECT")
            reason = ""
            if "Reason:" in raw:
                reason = raw.split("Reason:")[-1].strip()[:200]
            
            time.sleep(0.3)
            return {"verdict": verdict, "reason": reason, "raw": raw[:300]}
            
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 30 * (attempt + 1)
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err or "UNAVAILABLE" in err.lower():
                time.sleep(15)
            else:
                print(f"    Judge error attempt {attempt+1}: {err[:100]}")
                time.sleep(5)
    
    return {"verdict": False, "reason": "JUDGE_FAILED", "raw": ""}


# ── Core processing ───────────────────────────────────────────────────────────

def process_model_csv(client, client_type: str, csv_path: Path,
                      out_path: Path) -> list[dict]:
    """Process one model CSV with judge fallback. Returns audit log entries."""
    
    rows = []
    fieldnames = None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    
    if not rows:
        print(f"  SKIP: empty CSV")
        return []
    
    # Resume from existing output
    done_ids: set[str] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                done_ids.add(row["id"])
        print(f"  Resuming: {len(done_ids)} already processed")
    
    # Identify items needing judge
    needs_judge = [
        r for r in rows
        if int(r.get("correct", "0") or "0") == 0
        and r.get("task_type", "") in JUDGE_TASKS
        and "FAIL" not in r.get("prediction", "")
        and r.get("prediction", "").strip() != ""
        and r["id"] not in done_ids
    ]
    
    total_in_csv = len(rows)
    already_done = len(done_ids)
    
    print(f"  Total rows: {total_in_csv} | "
          f"Needs judge: {len(needs_judge)} | "
          f"Already done: {already_done}")
    
    audit_log: list[dict] = []
    output_rows: list[dict] = list(existing_rows)
    
    # Pass-through rows that don't need judge
    for row in rows:
        if row["id"] in done_ids:
            continue
        
        task = row.get("task_type", "")
        auto_score = int(row.get("correct", "0") or "0")
        
        is_fail = "FAIL" in row.get("prediction", "") or row.get("prediction", "").strip() == ""
        needs_j = (auto_score == 0 and task in JUDGE_TASKS and not is_fail)
        
        if needs_j:
            result = call_judge(
                client, client_type,
                question=row.get("question", ""),
                reference=row.get("ref_answer", row.get("reference_answer", "")),
                prediction=row.get("prediction", ""),
            )
            
            new_score = 1 if result["verdict"] else 0
            
            new_row = dict(row)
            new_row["auto_score"] = str(auto_score)
            new_row["judge_score"] = str(new_score)
            new_row["judge_reason"] = result["reason"]
            new_row["correct"] = str(new_score)
            
            if new_score == 1:
                ts = TASK_ABBR.get(task, task[:3])
                print(f"    FLIPPED → 1  [{ts}] {row['id']}: "
                      f"{result['reason'][:70]}")
            
            audit_log.append({
                "id":           row["id"],
                "task_type":    task,
                "auto_score":   str(auto_score),
                "judge_score":  str(new_score),
                "flipped":      str(new_score != auto_score),
                "judge_reason": result["reason"],
                "question":     row.get("question", "")[:100],
                "ref_answer":   row.get("ref_answer", "")[:80],
                "prediction":   row.get("prediction", "")[:80],
            })
        else:
            new_row = dict(row)
            new_row["auto_score"] = str(auto_score)
            new_row["judge_score"] = ""
            new_row["judge_reason"] = ""
        
        output_rows.append(new_row)
        done_ids.add(row["id"])
        
        # Checkpoint every 20 items
        if len(output_rows) % 20 == 0 and len(output_rows) > already_done:
            _save_csv(output_rows, out_path)
    
    _save_csv(output_rows, out_path)
    return audit_log


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    
    base = ["id", "task_type", "difficulty", "question",
            "ref_answer", "prediction", "correct", "model_version",
            "auto_score", "judge_score", "judge_reason"]
    extra = [k for k in rows[0].keys() if k not in base]
    fields = base + extra
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy(csv_path: Path) -> dict:
    """Compute per-task accuracy from a result CSV."""
    task_scores: dict[str, list[int]] = defaultdict(list)
    overall: list[int] = []
    
    if not csv_path.exists():
        return {}
    
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "FAIL" in row.get("prediction", ""):
                continue
            c = int(row.get("correct", "0") or "0")
            t = TASK_ABBR.get(row.get("task_type", ""), "?")
            task_scores[t].append(c)
            overall.append(c)
    
    result = {t: round(sum(s)/len(s)*100, 1) for t, s in task_scores.items() if s}
    result["Overall"] = round(sum(overall)/len(overall)*100, 1) if overall else 0
    result["n"] = len(overall)
    return result


def print_comparison_table(summaries: list[dict]) -> None:
    """Print before/after comparison table."""
    print(f"\n{'='*80}")
    print("  CORRECTED TABLE 6 — Use these numbers in your paper")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'REG_B':>6} {'REG_A':>6} {'NUM_B':>6} {'NUM_A':>6} "
          f"{'CON':>6} {'TMP_B':>6} {'TMP_A':>6} {'OV_B':>6} {'OV_A':>6} {'Δ':>5}")
    print(f"  {'-'*80}")
    
    for s in sorted(summaries, key=lambda x: -x.get("after_overall", 0)):
        print(f"  {s['model']:<25} "
              f"{s.get('before_REG',0):>5.1f}% "
              f"{s.get('after_REG',0):>5.1f}% "
              f"{s.get('before_NUM',0):>5.1f}% "
              f"{s.get('after_NUM',0):>5.1f}% "
              f"{s.get('after_CON',0):>5.1f}% "
              f"{s.get('before_TMP',0):>5.1f}% "
              f"{s.get('after_TMP',0):>5.1f}% "
              f"{s.get('before_overall',0):>5.1f}% "
              f"{s.get('after_overall',0):>5.1f}% "
              f"{s.get('delta_overall',0):>+4.1f}%")
    
    print(f"\n  OV_B = Overall before correction (current Table 6)")
    print(f"  OV_A = Overall after correction (NEW Table 6)")
    print(f"  _B/_A = before/after per task")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IndiaFinBench hybrid scoring with Gemini judge"
    )
    parser.add_argument("--dataset",
                        default="annotation/raw_qa/indiafinbench_qa_combined_406.json")
    parser.add_argument("--results_dir", default="evaluation/results")
    parser.add_argument("--output",      default="evaluation/results_judged")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model keys (default: all found)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*65}")
    print(f"  IndiaFinBench — Hybrid Scoring Pipeline")
    print(f"  Judge: Gemini 2.5 Flash via Vertex AI (GCP credits)")
    print(f"  Tasks being judged: NUM, REG, TMP")
    print(f"  CON: exact Yes/No match only (no judge needed)")
    print(f"{'='*65}\n")
    
    # Initialize Gemini client
    client, client_type = get_gemini_client()
    
    # Find which model CSVs exist
    models_to_run: list[tuple[str, Path]] = []
    
    if args.models:
        for mk in args.models:
            fname = MODEL_FILES.get(mk, f"{mk}_results.csv")
            path = results_dir / fname
            if path.exists():
                models_to_run.append((mk, path))
            else:
                print(f"  MISSING: {path}")
    else:
        for mk, fname in MODEL_FILES.items():
            path = results_dir / fname
            if path.exists():
                models_to_run.append((mk, path))
    
    if not models_to_run:
        print(f"No result CSVs found in {results_dir}")
        print("Make sure your evaluation/results/ directory has *_results.csv files")
        return
    
    print(f"Found {len(models_to_run)} model(s) to process:\n")
    for mk, path in models_to_run:
        print(f"  {MODEL_LABELS.get(mk, mk)}: {path.name}")
    print()
    
    all_audit: list[dict] = []
    summaries: list[dict] = []
    
    for mk, csv_path in models_to_run:
        label = MODEL_LABELS.get(mk, mk)
        out_path = output_dir / csv_path.name
        
        print(f"\n── {label} ──")
        
        before = compute_accuracy(csv_path)
        audit  = process_model_csv(client, client_type, csv_path, out_path)
        after  = compute_accuracy(out_path)
        
        all_audit.extend([{**a, "model": label} for a in audit])
        
        summaries.append({
            "model":          label,
            "before_overall": before.get("Overall", 0),
            "after_overall":  after.get("Overall", 0),
            "delta_overall":  round(after.get("Overall", 0) - before.get("Overall", 0), 1),
            "before_REG":     before.get("REG", 0),
            "after_REG":      after.get("REG", 0),
            "before_NUM":     before.get("NUM", 0),
            "after_NUM":      after.get("NUM", 0),
            "after_CON":      after.get("CON", 0),
            "before_TMP":     before.get("TMP", 0),
            "after_TMP":      after.get("TMP", 0),
            "n":              after.get("n", 0),
        })
        
        delta = after.get("Overall", 0) - before.get("Overall", 0)
        print(f"  → Overall: {before.get('Overall',0):.1f}% → "
              f"{after.get('Overall',0):.1f}% ({delta:+.1f}pp)")
    
    # Save audit log
    if all_audit:
        audit_path = output_dir / "judge_audit_log.csv"
        with open(audit_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_audit[0].keys()))
            w.writeheader()
            w.writerows(all_audit)
        print(f"\nAudit log: {audit_path}")
        
        # Count flips
        flipped = sum(1 for a in all_audit if a.get("flipped") == "True")
        print(f"Total items flipped 0→1: {flipped} of {len(all_audit)} judged")
    
    # Save comparison CSV
    if summaries:
        comp_path = output_dir / "pipeline_comparison.csv"
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"Comparison table: {comp_path}")
    
    # Print final table
    print_comparison_table(summaries)
    
    print(f"\n{'='*65}")
    print(f"  NEXT STEPS:")
    print(f"  1. Use 'after' numbers as your new Table 6")
    print(f"  2. The 'before' numbers go into Appendix D as 'automated pipeline'")
    print(f"  3. Section 4.3: mention hybrid pipeline with Gemini judge")
    print(f"  4. Section 5.1: remove FNR disclosure, replace with pipeline note")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
