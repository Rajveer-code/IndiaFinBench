"""
Experiment 10: Perturbation Robustness
Creates 3 variants of 60 items and tests Gemini 2.5 Flash + LLaMA-3.3-70B.
Measures robustness = % of answers unchanged under equivalent perturbations.

Field fix: dataset uses 'answer' (not 'reference_answer').
"""
import sys, re, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, call_gemini, call_groq, OUTPUT_DIR, FIGURES_DIR, TASK_MAP
)
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT = OUTPUT_DIR / "perturbation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def perturb_date_format(text):
    """Shift years by +5 (e.g., 2019 → 2024)."""
    def shift_year(m):
        year = int(m.group(0))
        if 1990 <= year <= 2025:
            return str(year + 5)
        return m.group(0)
    return re.sub(r'\b(19[89]\d|20[0-2]\d)\b', shift_year, text)


def perturb_number_format(text):
    """Change crore → million (10 crore ≈ 100 million)."""
    return re.sub(
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*crore',
        lambda m: f"{float(m.group(1).replace(',', '')) * 10:.0f} million",
        text, flags=re.IGNORECASE
    )


def perturb_amendment_synonyms(text):
    """Replace amendment keywords with synonyms."""
    replacements = {
        r'\bsupersedes\b': 'replaces',
        r'\bsuperseded\b': 'replaced',
        r'\bin supersession of\b': 'in replacement of',
        r'\bhereby\b': 'as a result',
        r'\bwith effect from\b': 'effective',
        r'\bamended\b': 'modified'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


EVAL_PROMPT = """You are an expert in Indian financial regulatory text.
Context: {context}
Question: {question}
Answer concisely. Give ONLY the answer, no explanation."""


def evaluate_item(context, question, reference, model_name="Gemini 2.5 Flash"):
    """Evaluate a single item with specified model."""
    prompt = EVAL_PROMPT.format(
        context=str(context)[:2000],
        question=str(question)
    )
    if 'Gemini' in model_name:
        pred = call_gemini(prompt)
    else:
        pred = call_groq(prompt)

    from rapidfuzz import fuzz
    ref_l = str(reference).lower().strip()
    pred_l = str(pred).lower().strip()
    exact = int(ref_l in pred_l or pred_l == ref_l)
    fuzzy = fuzz.token_set_ratio(ref_l, pred_l) / 100.0
    return pred, int(exact or fuzzy >= 0.72)


def main():
    print("Loading dataset...")
    dataset = load_dataset()

    sample_items = []
    for task in dataset['task_type'].unique():
        task_items = dataset[dataset['task_type'] == task]
        n = min(15, len(task_items))
        sample_items.append(task_items.sample(n, random_state=42))
    sample_df = pd.concat(sample_items).reset_index(drop=True)

    print(f"Running perturbation tests on {len(sample_df)} items...")

    cache_path = OUTPUT / "perturbation_results.json"
    if cache_path.exists():
        print("Loading cached results...")
        with open(cache_path) as f:
            all_results = json.load(f)
    else:
        all_results = []
        models_to_test = ["Gemini 2.5 Flash"]

        for model_name in models_to_test:
            print(f"\nModel: {model_name}")
            for i, (idx, item) in enumerate(sample_df.iterrows()):
                print(f"  Item {i+1}/{len(sample_df)}")
                ctx = str(item.get('context', '') or '')
                q = str(item.get('question', '') or '')
                # Use 'answer' field (verified dataset field name)
                ref = str(item.get('answer', '') or '')
                task = str(item.get('task_type', '') or '')

                pred_orig, corr_orig = evaluate_item(ctx, q, ref, model_name)
                pred_a, corr_a = evaluate_item(perturb_date_format(ctx), perturb_date_format(q), ref, model_name)
                pred_b, corr_b = evaluate_item(perturb_number_format(ctx), q, ref, model_name)
                pred_c, corr_c = evaluate_item(perturb_amendment_synonyms(ctx), q, ref, model_name)

                consistency_a = int(pred_orig.strip()[:50] == pred_a.strip()[:50])
                consistency_b = int(pred_orig.strip()[:50] == pred_b.strip()[:50])
                consistency_c = int(pred_orig.strip()[:50] == pred_c.strip()[:50])

                all_results.append({
                    'item_idx': int(idx),
                    'model': model_name,
                    'task_type': task,
                    'original_correct': corr_orig,
                    'date_perturb_correct': corr_a,
                    'number_perturb_correct': corr_b,
                    'synonym_perturb_correct': corr_c,
                    'consistency_date': consistency_a,
                    'consistency_number': consistency_b,
                    'consistency_synonym': consistency_c,
                    'robustness_score': (consistency_a + consistency_b + consistency_c) / 3
                })

                if (i + 1) % 5 == 0:
                    with open(cache_path, 'w') as f:
                        json.dump(all_results, f, indent=2)

        with open(cache_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT / "perturbation_results.csv", index=False)

    print("\nRobustness Analysis:")
    print(f"  Overall robustness: {results_df['robustness_score'].mean():.1%}")
    print(f"  Date perturbation: {results_df['consistency_date'].mean():.1%} answers unchanged")
    print(f"  Number perturbation: {results_df['consistency_number'].mean():.1%} answers unchanged")
    print(f"  Synonym perturbation: {results_df['consistency_synonym'].mean():.1%} answers unchanged")

    print(f"\nAccuracy under perturbations:")
    print(f"  Original: {results_df['original_correct'].mean():.1%}")
    print(f"  After date shift: {results_df['date_perturb_correct'].mean():.1%}")
    print(f"  After number format: {results_df['number_perturb_correct'].mean():.1%}")
    print(f"  After synonym change: {results_df['synonym_perturb_correct'].mean():.1%}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    conditions = ['Original', 'Date Shift', 'Number Format', 'Synonym']
    acc_vals = [results_df['original_correct'].mean(),
                results_df['date_perturb_correct'].mean(),
                results_df['number_perturb_correct'].mean(),
                results_df['synonym_perturb_correct'].mean()]
    colors = ['#4CAF50', '#F44336', '#FF9800', '#2196F3']
    bars = ax.bar(conditions, acc_vals, color=colors, alpha=0.85, edgecolor='white')
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy Under Perturbation\n(Gemini 2.5 Flash)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.1%}',
                ha='center', fontsize=10)

    ax2 = axes[1]
    if 'task_type' in results_df.columns:
        task_rob = results_df.groupby('task_type')['robustness_score'].mean()
        task_labels = [TASK_MAP.get(t, t[:3]) for t in task_rob.index]
        ax2.bar(task_labels, task_rob.values, color=['#2196F3'] * len(task_rob), alpha=0.85)
        ax2.set_ylabel('Robustness Score', fontsize=11)
        ax2.set_title('Perturbation Robustness by Task\n(1.0 = fully consistent)',
                      fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp10_perturbation.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print("=== EXPERIMENT 10 COMPLETE ===")


if __name__ == "__main__":
    main()
