"""
Experiment 5: Context Length vs Accuracy Analysis
Tests whether poor performance is caused by context length rather than reasoning ability.
"""
import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, load_correctness_matrix,
    OUTPUT_DIR, FIGURES_DIR, MODEL_FILES, _correctness_col
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT = OUTPUT_DIR / "context_length"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    dataset = load_dataset()
    all_results = load_all_results()

    # Context length — dataset has 'context' field (verified)
    ctx_col = 'context'
    if ctx_col not in dataset.columns:
        ctx_col = [c for c in dataset.columns if 'context' in c.lower()][0]

    dataset['ctx_tokens'] = dataset[ctx_col].apply(lambda x: len(str(x or '').split()))
    dataset['ctx_chars'] = dataset[ctx_col].apply(lambda x: len(str(x or '')))

    print(f"Context length stats:")
    print(f"  Mean: {dataset['ctx_tokens'].mean():.0f} tokens")
    print(f"  Min: {dataset['ctx_tokens'].min()}, Max: {dataset['ctx_tokens'].max()}")
    print(f"  Quartiles: {dataset['ctx_tokens'].quantile([.25, .5, .75]).to_dict()}")

    dataset['length_bucket'] = pd.qcut(
        dataset['ctx_tokens'], q=4,
        labels=['Q1 (Short)', 'Q2', 'Q3', 'Q4 (Long)'],
        duplicates='drop'
    )

    key_models = list(MODEL_FILES.keys())
    corr_results = []

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, model_name in enumerate(key_models[:6]):
        if model_name not in all_results:
            continue

        res_df = all_results[model_name]
        corr_col_name = _correctness_col(res_df)
        n = min(len(res_df), len(dataset))

        ctx_lengths = dataset['ctx_tokens'].values[:n]
        correctness = res_df[corr_col_name].values[:n]

        r, p = stats.pearsonr(ctx_lengths, correctness)
        corr_results.append({'model': model_name, 'pearson_r': round(r, 3), 'p_value': round(p, 4)})

        ax = axes[i]
        ax.scatter(ctx_lengths,
                   correctness + np.random.default_rng(0).normal(0, 0.02, n),
                   alpha=0.3, s=20, c=correctness, cmap='RdYlGn')

        bins = np.linspace(ctx_lengths.min(), ctx_lengths.max(), 10)
        binned_acc, bin_centers = [], []
        for j in range(len(bins) - 1):
            mask = (ctx_lengths >= bins[j]) & (ctx_lengths < bins[j + 1])
            if mask.sum() > 3:
                binned_acc.append(correctness[mask].mean())
                bin_centers.append((bins[j] + bins[j + 1]) / 2)

        if bin_centers:
            ax.plot(bin_centers, binned_acc, 'b-o', linewidth=2, markersize=5, label='Binned avg')

        ax.set_title(f'{model_name}\nr={r:.3f}, p={p:.3f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Context Length (tokens)', fontsize=8)
        ax.set_ylabel('Correct (0/1)', fontsize=8)
        ax.set_ylim(-0.1, 1.2)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle('Context Length vs Accuracy by Model', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = FIGURES_DIR / "exp5_context_length.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    pd.DataFrame(corr_results).to_csv(OUTPUT / "context_length_correlations.csv", index=False)
    print("\nContext length correlations:")
    print(pd.DataFrame(corr_results).to_string(index=False))
    print("=== EXPERIMENT 5 COMPLETE ===")


if __name__ == "__main__":
    main()
