"""
run_ai_annotator.py
--------------------
Uses LLaMA-3.3-70B via Groq as a second annotator for IndiaFinBench.
Strictly constrained to use only the provided context passage.

Model: llama-3.3-70b-versatile (NOT in the evaluation set — methodologically valid)
Cost: Free on Groq's free tier

Usage:
    set GROQ_API_KEY=your_key_here
    python scripts/run_ai_annotator.py

Outputs:
    annotation/annotated/ai_annotator_answers.csv

Then compute Kappa:
    python scripts/compute_kappa.py \
        --ref  annotation/raw_qa/indiafinbench_qa_combined_150.json \
        --ann2 annotation/annotated/ai_annotator_answers.csv \
        --out  annotation/inter_annotator/kappa_report.csv
"""

import json
import csv
import os
import time
import sys
from groq import Groq

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL       = "llama-3.3-70b-versatile"
QA_PATH     = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
OUTPUT_PATH = "annotation/annotated/ai_annotator_answers.csv"
DELAY_SEC   = 0.3   # polite delay between calls

# ── System prompt — strict context-only constraint ─────────────────────────────

SYSTEM_PROMPT = """You are an annotation assistant for a research benchmark dataset.
Your ONLY job is to answer questions using the provided context passage.

ABSOLUTE RULES — do not break any of these:
1. Use ONLY information present in the context passage. Never use outside knowledge.
2. If the answer cannot be found in the passage, write exactly: Cannot be determined from context
3. Give only the answer itself. No explanations, no disclaimers, no commentary.
4. Copy exact phrases from the context where possible. Do not paraphrase unnecessarily.
5. Keep answers concise — one sentence or less for most questions."""

# ── Task-specific prompts ──────────────────────────────────────────────────────

def build_prompt(item: dict) -> str:
    task = item["task_type"]
    q    = item["question"]

    if task == "contradiction_detection":
        return (
            f"Passage A:\n{item['context_a']}\n\n"
            f"Passage B:\n{item['context_b']}\n\n"
            f"Question: {q}\n\n"
            f"Answer (write 'Yes' or 'No' first, then one sentence of explanation "
            f"using only the two passages above):"
        )
    elif task == "numerical_reasoning":
        return (
            f"Context passage:\n{item['context']}\n\n"
            f"Question: {q}\n\n"
            f"Answer (show your calculation and final answer with units, "
            f"using only numbers from the context passage):"
        )
    elif task == "temporal_reasoning":
        return (
            f"Context passage:\n{item['context']}\n\n"
            f"Question: {q}\n\n"
            f"Answer (pay close attention to amendment dates, 'prior to', "
            f"and 'w.e.f.' language in the context):"
        )
    else:  # regulatory_interpretation
        return (
            f"Context passage:\n{item['context']}\n\n"
            f"Question: {q}\n\n"
            f"Answer (using only the context passage above):"
        )


# ── Single item annotation ─────────────────────────────────────────────────────

def annotate_item(client: Groq, item: dict, retries: int = 3) -> str:
    prompt = build_prompt(item)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=200,
                temperature=0.0,   # deterministic — important for reproducibility
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt == retries - 1:
                return f"API_ERROR: {str(e)[:80]}"
            wait = (attempt + 1) * 5
            print(f"      Retry {attempt+1}/{retries} in {wait}s — {e}")
            time.sleep(wait)

    return "API_ERROR"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Check API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nERROR: GROQ_API_KEY environment variable not set.")
        print("Run this first:")
        print("  Windows:  set GROQ_API_KEY=your_key_here")
        print("  Mac/Linux: export GROQ_API_KEY=your_key_here\n")
        sys.exit(1)

    # Load QA data
    if not os.path.exists(QA_PATH):
        print(f"\nERROR: QA file not found at {QA_PATH}")
        print("Make sure you are running this from inside IndiaFinBench/\n")
        sys.exit(1)

    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Check for existing checkpoint
    existing = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("your_answer") and "API_ERROR" not in row["your_answer"]:
                    existing[row["id"]] = row["your_answer"]
        if existing:
            print(f"\n  Checkpoint found — {len(existing)} items already done. Resuming.")

    client   = Groq(api_key=api_key)
    results  = []
    errors   = 0
    skipped  = 0

    print(f"\n{'━'*62}")
    print(f"  IndiaFinBench — AI Second Annotator")
    print(f"  Model  : {MODEL} via Groq")
    print(f"  Items  : {len(data)}")
    print(f"  Constraint: context-only (temperature=0, strict system prompt)")
    print(f"{'━'*62}\n")

    for i, item in enumerate(data, 1):
        item_id   = item["id"]
        task_type = item["task_type"]

        # Skip if already done in a previous run
        if item_id in existing:
            results.append({
                "id":          item_id,
                "task_type":   task_type,
                "difficulty":  item.get("difficulty", ""),
                "your_answer": existing[item_id],
            })
            skipped += 1
            continue

        answer = annotate_item(client, item)

        if "API_ERROR" in answer:
            errors += 1
            status = "✗ ERROR"
        else:
            status = "✓"

        results.append({
            "id":          item_id,
            "task_type":   task_type,
            "difficulty":  item.get("difficulty", ""),
            "your_answer": answer,
        })

        print(f"  [{i:03d}/150]  {item_id:<12}  {task_type[:28]:<28}  {status}")

        # Save checkpoint every 25 items
        if i % 25 == 0:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["id","task_type","difficulty","your_answer"]
                )
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  💾  Checkpoint saved at item {i} — {errors} errors so far\n")

        time.sleep(DELAY_SEC)

    # Final save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id","task_type","difficulty","your_answer"]
        )
        writer.writeheader()
        writer.writerows(results)

    successful = len(results) - errors - skipped
    print(f"\n{'━'*62}")
    print(f"  ✅  Annotation complete")
    print(f"  Successful : {successful + skipped}")
    print(f"  Errors     : {errors}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"\n  Next — compute Kappa:")
    print(f"  python scripts/compute_kappa.py \\")
    print(f"    --ref  {QA_PATH} \\")
    print(f"    --ann2 {OUTPUT_PATH} \\")
    print(f"    --out  annotation/inter_annotator/kappa_report.csv")
    print(f"{'━'*62}\n")


if __name__ == "__main__":
    main()