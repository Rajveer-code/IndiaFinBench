"""
Experiment 11: Structural Analyses
- Kendall's W concordance across model task-accuracy rankings
- Era stratification (pre-2000, 2000-2015, post-2015 documents)
- Consensus-hard item identification

Field fix: use 'document' column (not 'source') for year extraction.
The 'source' column contains SEBI/RBI names; 'document' contains filenames with years.
"""
import sys, json, re
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, get_task_accuracies, load_correctness_matrix,
    OUTPUT_DIR, FIGURES_DIR, _correctness_col
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata
import pingouin as pg

OUTPUT = OUTPUT_DIR / "era_stratification"
OUTPUT.mkdir(parents=True, exist_ok=True)
KENDALLS_W_DIR = OUTPUT_DIR / "kendalls_w"
KENDALLS_W_DIR.mkdir(parents=True, exist_ok=True)


def compute_kendalls_w(rankings_matrix):
    """Compute Kendall's W from rankings matrix (models × tasks)."""
    n_raters, n_items = rankings_matrix.shape
    ranks = np.array([rankdata(row) for row in rankings_matrix])
    S = np.sum((ranks.sum(axis=0) - ranks.sum(axis=0).mean()) ** 2)
    W = 12 * S / (n_raters ** 2 * (n_items ** 3 - n_items))
    chi2 = n_raters * (n_items - 1) * W
    df = n_items - 1
    p_value = float(1 - stats.chi2.cdf(chi2, df))
    return float(W), float(chi2), p_value


def extract_doc_year(source_str):
    """Extract year from document filename (e.g., SEBI_2018_securities...txt).

    Uses (?<!\\d)/(?!\\d) instead of \\b because filenames use underscores
    (word characters) as separators, preventing \\b from firing.
    """
    if pd.isna(source_str) or source_str != source_str:
        return np.nan
    years = re.findall(r'(?<!\d)(19[89]\d|20[0-2]\d)(?!\d)', str(source_str))
    return int(years[0]) if years else np.nan


def main():
    print("Loading data...")
    task_accs = get_task_accuracies()
    dataset = load_dataset()
    all_results = load_all_results()
    matrix = load_correctness_matrix()

    # ── Kendall's W ───────────────────────────────────────────────────────
    print("\n--- Kendall's W Analysis ---")
    task_order = ['REG', 'NUM', 'CON', 'TMP']
    available_tasks = [t for t in task_order if t in task_accs.columns]
    rankings_data = task_accs[available_tasks].fillna(0).values

    W, chi2, p = compute_kendalls_w(rankings_data)
    print(f"Kendall's W = {W:.3f} (chi2={chi2:.2f}, df={len(available_tasks)-1}, p={p:.4f})")
    strength = 'Strong' if W > 0.7 else 'Moderate' if W > 0.4 else 'Weak'
    print(f"Interpretation: {strength} concordance")
    print("(High W = models agree on which tasks are hard/easy)")

    with open(KENDALLS_W_DIR / "kendalls_w_results.json", 'w') as f:
        json.dump({
            'W': round(W, 3), 'chi2': round(chi2, 3), 'p_value': round(p, 4),
            'n_models': len(task_accs), 'n_tasks': len(available_tasks),
            'strength': strength
        }, f, indent=2)

    # ── Era Stratification ────────────────────────────────────────────────
    print("\n--- Era Stratification ---")

    # Use 'document' column (contains filenames with years, e.g., SEBI_2018_...)
    # 'source' only contains SEBI/RBI labels without years
    year_pat = r'(?<!\d)(19[89]\d|20[0-2]\d)(?!\d)'
    if 'document' in dataset.columns:
        dataset['doc_year'] = dataset['document'].apply(extract_doc_year)
    elif 'regulation' in dataset.columns:
        dataset['doc_year'] = dataset['regulation'].apply(extract_doc_year)
    else:
        # Last resort: extract from context
        dataset['doc_year'] = dataset['context'].apply(
            lambda x: int(re.findall(year_pat, str(x))[0])
            if re.findall(year_pat, str(x)) else np.nan
        )

    year_coverage = dataset['doc_year'].notna().sum()
    print(f"Year extracted for {year_coverage}/{len(dataset)} items")

    dataset['era'] = pd.cut(
        dataset['doc_year'],
        bins=[0, 1999, 2015, 2100],
        labels=['Pre-2000', '2000-2015', 'Post-2015']
    )
    era_counts = dataset['era'].value_counts()
    print(f"Era distribution: {era_counts.to_dict()}")

    era_results = []
    for era in ['Pre-2000', '2000-2015', 'Post-2015']:
        era_mask = dataset['era'] == era
        if era_mask.sum() == 0:
            continue
        era_indices = dataset[era_mask].index.tolist()
        for model_name, res_df in all_results.items():
            corr_c = _correctness_col(res_df)
            valid_idx = [i for i in era_indices if i < len(res_df)]
            era_acc = res_df.iloc[valid_idx][corr_c].mean() if valid_idx else np.nan
            era_results.append({
                'era': era, 'model': model_name,
                'accuracy': era_acc, 'n_items': int(era_mask.sum())
            })

    era_df = pd.DataFrame(era_results)
    era_df.to_csv(OUTPUT / "era_accuracy.csv", index=False)

    # ── Consensus-Hard Items ──────────────────────────────────────────────
    print("\n--- Consensus-Hard Item Analysis ---")
    item_accuracy = matrix.mean(axis=1)
    consensus_hard = item_accuracy[item_accuracy < 0.2]
    consensus_easy = item_accuracy[item_accuracy > 0.9]

    print(f"Consensus-hard items (< 20% models correct): {len(consensus_hard)}")
    print(f"Consensus-easy items (> 90% models correct): {len(consensus_easy)}")

    # ── Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Task ranking heatmap
    ax = axes[0]
    try:
        rank_df = pd.DataFrame(
            [rankdata(task_accs.loc[m, available_tasks].fillna(0).values)
             for m in task_accs.index],
            index=task_accs.index,
            columns=available_tasks
        )
        im = ax.imshow(rank_df.values, cmap='RdYlGn', aspect='auto', vmin=1, vmax=len(available_tasks))
        ax.set_xticks(range(len(available_tasks)))
        ax.set_xticklabels(available_tasks)
        ax.set_yticks(range(len(rank_df)))
        ax.set_yticklabels([str(m).split()[0] for m in rank_df.index], fontsize=7)
        plt.colorbar(im, ax=ax, label='Task Rank (1=best)')
        ax.set_title(f"Task Difficulty Rankings\n(Kendall's W={W:.3f}, p={p:.3f})",
                     fontsize=10, fontweight='bold')
    except Exception as e:
        print(f"  Rank heatmap error: {e}")

    # 2. Era accuracy bar
    ax2 = axes[1]
    if len(era_df) > 0:
        era_summary = era_df.groupby('era')['accuracy'].agg(['mean', 'std']).reset_index()
        eras_sorted = ['Pre-2000', '2000-2015', 'Post-2015']
        era_summary = era_summary.set_index('era').reindex(eras_sorted).dropna()
        if len(era_summary) > 0:
            ax2.bar(era_summary.index, era_summary['mean'],
                    yerr=era_summary['std'], capsize=5,
                    color=['#F44336', '#FF9800', '#4CAF50'], alpha=0.85, edgecolor='white')
            ax2.set_ylabel('Average Accuracy', fontsize=11)
            ax2.set_title('Accuracy by Document Era\n(tests temporal generalization)',
                          fontsize=11, fontweight='bold')
            ax2.set_ylim(0, 1.1)
            ax2.grid(axis='y', alpha=0.3)

    # 3. Consensus difficulty histogram
    ax3 = axes[2]
    ax3.hist(item_accuracy, bins=20, color='#2196F3', alpha=0.8, edgecolor='white')
    ax3.axvline(x=0.2, color='red', linestyle='--', linewidth=2,
                label=f'Hard threshold\n(n={len(consensus_hard)})')
    ax3.axvline(x=0.9, color='green', linestyle='--', linewidth=2,
                label=f'Easy threshold\n(n={len(consensus_easy)})')
    ax3.set_xlabel('Fraction of Models Correct', fontsize=11)
    ax3.set_ylabel('Number of Items', fontsize=11)
    ax3.set_title('Item Difficulty Distribution\n(Ensemble-based)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp11_structural.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print("=== EXPERIMENT 11 COMPLETE ===")


if __name__ == "__main__":
    main()
