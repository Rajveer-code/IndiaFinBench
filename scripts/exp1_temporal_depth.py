"""
Experiment 1: Temporal Chain Depth Analysis
For each TMP item, compute amendment chain complexity features.
Then plot accuracy vs complexity for all models.
"""
import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, OUTPUT_DIR, FIGURES_DIR, TASK_MAP
)
import re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT = OUTPUT_DIR / "tmp_depth"
OUTPUT.mkdir(parents=True, exist_ok=True)

AMENDMENT_KEYWORDS = [
    r'\bsupersed',
    r'\bamend',
    r'\bmodif',
    r'\breplace',
    r'\bsubstitut',
    r'\bhereby circu',
    r'\bin supersession',
    r'\bin partial modification',
    r'\bwith effect from',
    r'\beffective from',
    r'\bprevious circular',
    r'\bearlier circular',
    r'\bprior circular',
]


def extract_complexity_features(row):
    """Extract temporal complexity features from a QA item."""
    context = str(row.get('context', '') or '')
    question = str(row.get('question', '') or '')
    combined = context + ' ' + question
    combined_lower = combined.lower()

    dates = re.findall(
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b\d{4}\b)',
        combined
    )
    num_dates = len(set(dates))

    amendment_count = sum(
        len(re.findall(kw, combined_lower))
        for kw in AMENDMENT_KEYWORDS
    )

    circular_refs = len(re.findall(
        r'\bcircular\b|\bnotification\b|\bdirective\b|\bregulation\b|\bmaster direction\b',
        combined_lower
    ))

    context_len = len(context.split())

    years = re.findall(r'\b(19[89]\d|20[0-2]\d)\b', combined)
    num_years = len(set(years))

    complexity_score = (
        0.3 * min(num_dates / 5, 1.0) +
        0.4 * min(amendment_count / 3, 1.0) +
        0.2 * min(num_years / 4, 1.0) +
        0.1 * min(circular_refs / 5, 1.0)
    )

    if complexity_score < 0.2:
        tier = "Low"
    elif complexity_score < 0.5:
        tier = "Medium"
    else:
        tier = "High"

    return {
        'num_dates': num_dates,
        'amendment_count': amendment_count,
        'circular_refs': circular_refs,
        'context_len': context_len,
        'num_years': num_years,
        'complexity_score': round(complexity_score, 3),
        'complexity_tier': tier
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset()

    tmp_mask = dataset['task_type'].str.contains('temporal', case=False, na=False)
    tmp_items = dataset[tmp_mask].copy()
    print(f"Found {len(tmp_items)} TMP items")

    if len(tmp_items) == 0:
        print("Available task_types:", dataset['task_type'].unique())
        return

    print(f"Computing complexity features for {len(tmp_items)} TMP items...")
    features = tmp_items.apply(extract_complexity_features, axis=1)
    features_df = pd.DataFrame(list(features))
    tmp_items = pd.concat([tmp_items.reset_index(drop=True), features_df], axis=1)

    tmp_items.to_csv(OUTPUT / "tmp_items_with_complexity.csv", index=False)
    print(f"Saved complexity features.")

    # ── Merge with model results ──────────────────────────────────────────
    print("\nLoading model results and merging...")
    all_results = load_all_results()

    from scripts.novel_methods_utils import _correctness_col, _task_col, _id_col
    for model_name, res_df in all_results.items():
        task_c = _task_col(res_df)
        corr_c = _correctness_col(res_df)
        tmp_res = res_df[res_df[task_c].str.contains('temporal', case=False, na=False)].copy()
        if len(tmp_res) == 0:
            print(f"  WARNING: No TMP results for {model_name}")
            continue
        if len(tmp_res) == len(tmp_items):
            tmp_items[f'correct_{model_name.replace(" ", "_")}'] = tmp_res[corr_c].values
        else:
            print(f"  Size mismatch for {model_name}: {len(tmp_res)} vs {len(tmp_items)}")

    tmp_items.to_csv(OUTPUT / "tmp_items_merged.csv", index=False)

    # ── Accuracy by complexity tier ───────────────────────────────────────
    print("\nComputing accuracy by complexity tier...")
    correct_cols = [c for c in tmp_items.columns if c.startswith('correct_')]
    model_names = [c.replace('correct_', '').replace('_', ' ') for c in correct_cols]

    tier_accuracy = {}
    for tier in ["Low", "Medium", "High"]:
        mask = tmp_items['complexity_tier'] == tier
        tier_accuracy[tier] = {'n': int(mask.sum())}
        for corr_col, mname in zip(correct_cols, model_names):
            if corr_col in tmp_items.columns:
                tier_accuracy[tier][mname] = tmp_items.loc[mask, corr_col].mean()

    tier_df = pd.DataFrame(tier_accuracy).T
    tier_df.to_csv(OUTPUT / "accuracy_by_tier.csv")
    print("\nAccuracy by complexity tier:")
    print(tier_df)

    # ── Spearman correlations ─────────────────────────────────────────────
    print("\nComputing Spearman correlations...")
    corr_results = []
    for corr_col, mname in zip(correct_cols, model_names):
        if corr_col not in tmp_items.columns:
            continue
        r, p = stats.spearmanr(
            tmp_items['complexity_score'],
            tmp_items[corr_col].fillna(0),
            nan_policy='omit'
        )
        corr_results.append({'model': mname, 'spearman_r': round(r, 3),
                              'p_value': round(p, 4), 'significant': p < 0.05})

    corr_df = pd.DataFrame(corr_results).sort_values('spearman_r')
    corr_df.to_csv(OUTPUT / "complexity_accuracy_correlations.csv", index=False)
    print(corr_df)

    # ── Plot 1: Accuracy vs Complexity Tier ───────────────────────────────
    print("\nGenerating plots...")
    key_models = ["Gemini 2.5 Flash", "Qwen3-32B", "LLaMA-3.3-70B",
                  "DeepSeek R1 70B", "Gemma 4 E4B"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    tiers = ["Low", "Medium", "High"]
    x = np.arange(len(tiers))
    width = 0.15
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for i, model in enumerate(key_models):
        y_vals = []
        for tier in tiers:
            val = tier_df.loc[tier, model] if (tier in tier_df.index and model in tier_df.columns) else 0
            y_vals.append(float(val) if not pd.isna(val) else 0)
        ax.bar(x + i * width, y_vals, width, label=model,
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Complexity Tier', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('TMP Accuracy by Amendment Chain Complexity', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(tiers)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    for j, tier in enumerate(tiers):
        n = tier_df.loc[tier, 'n'] if ('n' in tier_df.columns and tier in tier_df.index) else '?'
        ax.text(j + width * 2, 1.02, f'n={int(n)}', ha='center', fontsize=9)

    ax2 = axes[1]
    top_model = key_models[0]
    top_col = f"correct_{top_model.replace(' ', '_')}"
    if top_col in tmp_items.columns:
        jitter = np.random.default_rng(0).normal(0, 0.02, len(tmp_items))
        ax2.scatter(tmp_items['complexity_score'],
                    tmp_items[top_col].fillna(0) + jitter,
                    c=tmp_items['complexity_score'], cmap='RdYlGn_r', alpha=0.6, s=50)
        x_vals = tmp_items['complexity_score'].values
        y_vals = tmp_items[top_col].fillna(0).values
        if len(x_vals) > 5:
            z = np.polyfit(x_vals, y_vals, 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax2.plot(x_line, p_line(x_line), 'r--', linewidth=2, label='Trend')
        ax2.set_xlabel('Complexity Score', fontsize=12)
        ax2.set_ylabel('Correct (1=Yes, 0=No)', fontsize=12)
        ax2.set_title(f'Accuracy vs. Complexity: {top_model}', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.1, 1.2)
        ax2.legend()
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp1_temporal_complexity.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")

    # ── Plot 2: Accuracy drop Low → High ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    all_mnames = [m for m in model_names if m in tier_df.columns]
    drops = []
    for model in all_mnames:
        low_acc = float(tier_df.loc['Low', model]) if ('Low' in tier_df.index and model in tier_df.columns) else 0
        high_acc = float(tier_df.loc['High', model]) if ('High' in tier_df.index and model in tier_df.columns) else 0
        drop = low_acc - high_acc
        drops.append((model, drop))

    drops.sort(key=lambda x: x[1], reverse=True)
    bar_colors = ['#F44336' if d[1] > 0.1 else '#FF9800' if d[1] > 0 else '#4CAF50' for d in drops]
    bars = ax.barh([d[0] for d in drops], [d[1] for d in drops],
                   color=bar_colors, alpha=0.85, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Accuracy Drop (Low → High Complexity)', fontsize=12)
    ax.set_title('Temporal Reasoning Sensitivity to Amendment Chain Complexity\n'
                 '(Positive = worse on complex items)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, (_, val) in zip(bars, drops):
        ax.text(val + 0.005 if val >= 0 else val - 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.1%}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.tight_layout()
    out_path2 = FIGURES_DIR / "exp1_accuracy_drop.png"
    plt.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path2}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'total_tmp_items': len(tmp_items),
        'tier_distribution': tmp_items['complexity_tier'].value_counts().to_dict(),
        'avg_complexity_score': round(float(tmp_items['complexity_score'].mean()), 4),
        'avg_num_dates': round(float(tmp_items['num_dates'].mean()), 2),
        'avg_amendment_count': round(float(tmp_items['amendment_count'].mean()), 2),
        'correlations': corr_df.to_dict('records')
    }
    with open(OUTPUT / "exp1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== EXPERIMENT 1 COMPLETE ===")
    if len(corr_df) > 0:
        print(f"Key finding: {corr_df.iloc[0].to_dict()}")
    print(f"Items per tier: {tmp_items['complexity_tier'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
