"""
Experiment 4: Scoring Pipeline False-Negative Audit
For 100 items marked INCORRECT by the automated pipeline,
use Gemini as semantic judge to estimate false-negative rate.

Field fix: results CSV uses 'ref_answer' (not 'reference_answer').
The _reference_col() helper handles this automatically.
"""
import sys, json, re, random
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_all_results, call_gemini, OUTPUT_DIR, FIGURES_DIR,
    _correctness_col, _task_col, _prediction_col, _reference_col
)
import matplotlib.pyplot as plt
import pandas as pd

OUTPUT = OUTPUT_DIR / "scoring_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

JUDGE_PROMPT = """You are an expert in Indian financial regulatory text.

Question: {question}
Reference Answer: {reference}
Model Prediction: {prediction}

Is the model prediction semantically equivalent to the reference answer,
taking into account Indian financial regulatory context (e.g., crore/lakh notation,
regulatory abbreviations, equivalent phrasings)?

Answer with ONLY one of:
CORRECT - if prediction is semantically equivalent or contains the reference answer
INCORRECT - if prediction gives wrong information
PARTIAL - if prediction is partially correct

Then give a one-line reason.

Format: VERDICT|reason"""


def main():
    print("Loading results...")
    all_results = load_all_results()

    incorrect_items = []
    for model_name, res_df in all_results.items():
        corr_c = _correctness_col(res_df)
        task_c = _task_col(res_df)
        pred_c = _prediction_col(res_df)
        ref_c = _reference_col(res_df)  # handles 'ref_answer' correctly

        if pred_c is None or ref_c is None:
            print(f"  Skipping {model_name}: missing prediction or reference column")
            print(f"  Available columns: {res_df.columns.tolist()}")
            continue

        incorrect = res_df[res_df[corr_c] == 0].copy()
        for _, row in incorrect.iterrows():
            incorrect_items.append({
                'model': model_name,
                'task_type': str(row.get(task_c, '')),
                'prediction': str(row.get(pred_c, '')),
                'reference': str(row.get(ref_c, '')),
                'question': str(row.get('question', ''))
            })

    print(f"Total incorrect predictions: {len(incorrect_items)}")

    random.seed(42)
    tasks = ['regulatory', 'numerical', 'contradiction', 'temporal']
    sample_per_task = 25
    sampled = []

    for task in tasks:
        task_items = [i for i in incorrect_items if task in i['task_type'].lower()]
        n = min(sample_per_task, len(task_items))
        if n > 0:
            sampled.extend(random.sample(task_items, n))

    remaining = 100 - len(sampled)
    if remaining > 0:
        others = [i for i in incorrect_items if i not in sampled]
        if others:
            sampled.extend(random.sample(others, min(remaining, len(others))))

    print(f"Evaluating {len(sampled)} incorrectly-scored items...")

    cache_path = OUTPUT / "audit_results.json"
    if cache_path.exists():
        print("Loading cached results...")
        with open(cache_path) as f:
            audit_results = json.load(f)
    else:
        audit_results = []
        for i, item in enumerate(sampled):
            print(f"  [{i+1}/{len(sampled)}] {item['model']} / {item['task_type'][:20]}")
            prompt = JUDGE_PROMPT.format(
                question=item['question'][:500],
                reference=item['reference'][:200],
                prediction=item['prediction'][:200]
            )
            response = call_gemini(prompt)
            verdict = 'UNKNOWN'
            reason = ''
            if '|' in response:
                parts = response.strip().split('|', 1)
                verdict = parts[0].strip().upper()
                reason = parts[1].strip() if len(parts) > 1 else ''
            elif 'CORRECT' in response.upper():
                verdict = 'CORRECT'
            elif 'INCORRECT' in response.upper():
                verdict = 'INCORRECT'
            elif 'PARTIAL' in response.upper():
                verdict = 'PARTIAL'

            audit_results.append({
                'model': item['model'],
                'task_type': item['task_type'],
                'prediction': item['prediction'][:100],
                'reference': item['reference'][:100],
                'llm_verdict': verdict,
                'reason': reason
            })
            if (i + 1) % 10 == 0:
                with open(cache_path, 'w') as f:
                    json.dump(audit_results, f, indent=2)

        with open(cache_path, 'w') as f:
            json.dump(audit_results, f, indent=2)

    results_df = pd.DataFrame(audit_results)
    false_negatives = results_df[results_df['llm_verdict'] == 'CORRECT']
    partial = results_df[results_df['llm_verdict'] == 'PARTIAL']
    true_incorrect = results_df[results_df['llm_verdict'] == 'INCORRECT']
    fn_rate = len(false_negatives) / len(results_df)

    print(f"\nAudit Results ({len(results_df)} items):")
    print(f"  False Negatives (marked wrong, actually correct): {len(false_negatives)} ({fn_rate:.1%})")
    print(f"  Partially correct: {len(partial)} ({len(partial)/len(results_df):.1%})")
    print(f"  Truly incorrect: {len(true_incorrect)} ({len(true_incorrect)/len(results_df):.1%})")

    task_fn = results_df.groupby('task_type').apply(
        lambda x: (x['llm_verdict'] == 'CORRECT').mean()
    ).reset_index()
    task_fn.columns = ['task_type', 'false_negative_rate']
    task_fn.to_csv(OUTPUT / "fn_rate_by_task.csv", index=False)
    print("\nFalse negative rate by task:")
    print(task_fn.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    counts = [len(false_negatives), len(partial), len(true_incorrect)]
    labels = [f'False Negative\n({fn_rate:.1%})',
              f'Partial\n({len(partial)/len(results_df):.1%})',
              f'Truly Wrong\n({len(true_incorrect)/len(results_df):.1%})']
    ax.pie(counts, labels=labels, colors=['#4CAF50', '#FF9800', '#F44336'],
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('Scoring Pipeline Audit\n(100 "incorrect" predictions re-evaluated)',
                 fontsize=11, fontweight='bold')

    ax2 = axes[1]
    ax2.bar(task_fn['task_type'], task_fn['false_negative_rate'],
            color='#2196F3', alpha=0.85, edgecolor='white')
    ax2.set_xlabel('Task Type', fontsize=11)
    ax2.set_ylabel('False Negative Rate', fontsize=11)
    ax2.set_title('False Negative Rate by Task\n(pipeline underestimation)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.5)
    ax2.grid(axis='y', alpha=0.3)
    for i, (_, row) in enumerate(task_fn.iterrows()):
        ax2.text(i, row['false_negative_rate'] + 0.01, f"{row['false_negative_rate']:.1%}",
                 ha='center', fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp4_scoring_audit.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    summary = {
        'total_audited': len(results_df),
        'false_negative_rate': round(fn_rate, 4),
        'partial_rate': round(len(partial) / len(results_df), 4),
        'true_incorrect_rate': round(len(true_incorrect) / len(results_df), 4),
        'by_task': task_fn.to_dict('records'),
        'implied_accuracy_correction': round(fn_rate * 0.5, 4)
    }
    with open(OUTPUT / "scoring_audit_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== EXPERIMENT 4 COMPLETE ===")
    print(f"Key finding: Pipeline false-negative rate = {fn_rate:.1%}")
    print(f"Implied true accuracy is ~{fn_rate:.1%} higher than reported scores.")


if __name__ == "__main__":
    main()
