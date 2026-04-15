"""
Experiment 2: Regulatory State Tracking Score (RSTS)
New metric that decomposes TMP answer quality into:
  - Event identification (did model find the right events?)
  - Temporal ordering (did model order them correctly?)
  - Final state (did model give the correct operative answer?)

Uses Gemini 2.5 Flash as judge (free tier) for a sample of 60 TMP items.

Field note: dataset answer field is 'answer', results ref field is 'ref_answer'.
"""
import sys, json, time, re
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, call_gemini,
    OUTPUT_DIR, FIGURES_DIR, _correctness_col, _task_col, _prediction_col
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT = OUTPUT_DIR / "rsts_scores"
OUTPUT.mkdir(parents=True, exist_ok=True)

RSTS_JUDGE_PROMPT = """You are an expert evaluator of Indian financial regulatory QA.

Given:
- CONTEXT: {context}
- QUESTION: {question}
- REFERENCE ANSWER: {reference}
- MODEL PREDICTION: {prediction}

Evaluate the model prediction on THREE dimensions. Answer ONLY with valid JSON:

{{
  "event_identification": {{
    "score": 0 or 1,
    "reason": "brief reason"
  }},
  "temporal_ordering": {{
    "score": 0 or 1,
    "reason": "brief reason (or N/A if no ordering required)"
  }},
  "final_state_answer": {{
    "score": 0 or 1,
    "reason": "brief reason"
  }}
}}

Scoring criteria:
- event_identification: 1 if model correctly identified the key regulatory events/dates/circulars relevant to answering
- temporal_ordering: 1 if model correctly placed events in chronological order (mark 1 if no ordering needed)
- final_state_answer: 1 if model's final answer matches or semantically equals the reference answer

Output ONLY the JSON object, no other text."""


def compute_rsts_scores(dataset, all_results, sample_size=60):
    """Compute RSTS for a sample of TMP items using LLM-as-judge."""
    tmp_mask = dataset['task_type'].str.contains('temporal', case=False, na=False)
    tmp_items = dataset[tmp_mask].copy().reset_index(drop=True)

    if len(tmp_items) == 0:
        print("ERROR: No TMP items found")
        return None

    if len(tmp_items) > sample_size:
        if 'difficulty' in tmp_items.columns:
            sample_per_diff = sample_size // 3
            sampled = []
            for diff in tmp_items['difficulty'].unique():
                diff_items = tmp_items[tmp_items['difficulty'] == diff]
                n = min(sample_per_diff, len(diff_items))
                sampled.append(diff_items.sample(n, random_state=42))
            tmp_sample = pd.concat(sampled).head(sample_size)
        else:
            tmp_sample = tmp_items.sample(sample_size, random_state=42)
    else:
        tmp_sample = tmp_items

    print(f"Computing RSTS for {len(tmp_sample)} TMP items...")

    models_to_eval = ["Gemini 2.5 Flash", "DeepSeek R1 70B", "LLaMA-3.3-70B"]
    all_rsts = []

    for model_name in models_to_eval:
        if model_name not in all_results:
            print(f"  Skipping {model_name} (no results)")
            continue

        res_df = all_results[model_name]
        task_c = _task_col(res_df)
        tmp_res = res_df[res_df[task_c].str.contains('temporal', case=False, na=False)].copy().reset_index(drop=True)
        pred_c = _prediction_col(res_df)
        corr_c = _correctness_col(res_df)

        print(f"\n  Processing {model_name} ({len(tmp_sample)} items)...")

        for i, (_, item) in enumerate(tmp_sample.iterrows()):
            if i >= len(tmp_res):
                continue
            prediction = str(tmp_res.iloc[i][pred_c]) if pred_c else "N/A"

            # Use 'answer' field (verified dataset field name)
            reference = str(item.get('answer', ''))

            prompt = RSTS_JUDGE_PROMPT.format(
                context=str(item.get('context', ''))[:1500],
                question=str(item.get('question', '')),
                reference=reference,
                prediction=prediction
            )

            response = call_gemini(prompt)

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    scores = json.loads(json_match.group())
                    rsts_entry = {
                        'item_idx': i,
                        'model': model_name,
                        'difficulty': item.get('difficulty', 'unknown'),
                        'event_score': scores.get('event_identification', {}).get('score', 0),
                        'ordering_score': scores.get('temporal_ordering', {}).get('score', 0),
                        'final_score': scores.get('final_state_answer', {}).get('score', 0),
                        'standard_correct': int(tmp_res.iloc[i][corr_c]) if i < len(tmp_res) else 0
                    }
                    rsts_entry['rsts'] = round(
                        0.25 * rsts_entry['event_score'] +
                        0.25 * rsts_entry['ordering_score'] +
                        0.50 * rsts_entry['final_score'],
                        3
                    )
                    all_rsts.append(rsts_entry)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    Parse error for item {i}: {e}")
                continue

        if all_rsts:
            pd.DataFrame(all_rsts).to_csv(OUTPUT / "rsts_scores_partial.csv", index=False)

    if not all_rsts:
        print("ERROR: No RSTS scores computed")
        return None

    rsts_df = pd.DataFrame(all_rsts)
    rsts_df.to_csv(OUTPUT / "rsts_scores_full.csv", index=False)
    return rsts_df


def generate_rsts_figures(rsts_df):
    """Generate RSTS comparison figures."""
    model_summary = rsts_df.groupby('model').agg({
        'rsts': 'mean',
        'standard_correct': 'mean',
        'event_score': 'mean',
        'ordering_score': 'mean',
        'final_score': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(len(model_summary))
    width = 0.35
    ax.bar(x - width / 2, model_summary['rsts'], width, label='RSTS (multi-dim)',
           color='#2196F3', alpha=0.85)
    ax.bar(x + width / 2, model_summary['standard_correct'], width,
           label='Standard Accuracy', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_summary['model']], fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RSTS vs. Standard Accuracy\non Temporal Reasoning Items',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    models = model_summary['model'].tolist()
    event_vals = model_summary['event_score'].values * 0.25
    ordering_vals = model_summary['ordering_score'].values * 0.25
    final_vals = model_summary['final_score'].values * 0.50
    ax2.bar(models, event_vals, label='Event ID (25%)', color='#4CAF50', alpha=0.85)
    ax2.bar(models, ordering_vals, bottom=event_vals, label='Ordering (25%)',
            color='#2196F3', alpha=0.85)
    ax2.bar(models, final_vals, bottom=event_vals + ordering_vals,
            label='Final State (50%)', color='#F44336', alpha=0.85)
    ax2.set_ylabel('RSTS Score (weighted)', fontsize=12)
    ax2.set_title('RSTS Component Breakdown\nby Model', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp2_rsts_scores.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    model_summary.to_csv(OUTPUT / "rsts_model_summary.csv", index=False)
    return model_summary


def main():
    print("Loading dataset and results...")
    dataset = load_dataset()
    all_results = load_all_results()

    rsts_cache = OUTPUT / "rsts_scores_full.csv"
    if rsts_cache.exists():
        print(f"Loading cached RSTS scores from {rsts_cache}")
        rsts_df = pd.read_csv(rsts_cache)
    else:
        rsts_df = compute_rsts_scores(dataset, all_results, sample_size=60)

    if rsts_df is None or len(rsts_df) == 0:
        print("ERROR: No RSTS data available")
        return

    print(f"\nRSTS scores computed for {len(rsts_df)} item-model pairs")
    model_summary = generate_rsts_figures(rsts_df)

    print("\n=== EXPERIMENT 2 COMPLETE ===")
    print("Model RSTS Summary:")
    print(model_summary[['model', 'rsts', 'standard_correct']].to_string(index=False))
    for _, row in model_summary.iterrows():
        diff = row['standard_correct'] - row['rsts']
        print(f"  {row['model']}: RSTS={row['rsts']:.1%} vs Accuracy={row['standard_correct']:.1%} (gap={diff:+.1%})")


if __name__ == "__main__":
    main()
