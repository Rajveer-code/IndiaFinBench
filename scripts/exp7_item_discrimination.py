"""
Experiment 7: IRT-lite Item Discrimination Analysis
Identifies items with highest discriminative power.
Validates that benchmark is not dominated by easy or redundant items.
"""
import sys, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_correctness_matrix, OUTPUT_DIR, FIGURES_DIR, TASK_MAP
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT = OUTPUT_DIR / "iRT_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading correctness matrix...")
    matrix = load_correctness_matrix()
    dataset = load_dataset()

    print(f"Matrix shape: {matrix.shape} (items × models)")

    item_stats = pd.DataFrame({
        'accuracy': matrix.mean(axis=1),
        'variance': matrix.var(axis=1),
        'n_correct': matrix.sum(axis=1),
        'n_total': matrix.notna().sum(axis=1)
    })

    item_stats['difficulty'] = 1 - item_stats['accuracy']
    item_stats['discrimination'] = 4 * item_stats['accuracy'] * (1 - item_stats['accuracy'])

    item_stats['category'] = 'Medium'
    item_stats.loc[item_stats['accuracy'] >= 0.9, 'category'] = 'Easy (ceiling)'
    item_stats.loc[item_stats['accuracy'] <= 0.1, 'category'] = 'Hard (floor)'
    item_stats.loc[
        (item_stats['accuracy'] > 0.3) & (item_stats['accuracy'] < 0.8) &
        (item_stats['discrimination'] > item_stats['discrimination'].median()),
        'category'
    ] = 'Highly Discriminative'

    if len(item_stats) == len(dataset):
        item_stats['task_type'] = dataset['task_type'].values
        item_stats['difficulty_label'] = dataset.get('difficulty',
                                          pd.Series(['unknown'] * len(dataset))).values

    item_stats.to_csv(OUTPUT / "item_discrimination.csv")

    print(f"\nItem categories:")
    print(item_stats['category'].value_counts().to_string())
    print(f"\nMost discriminative items (top 10):")
    top_disc = item_stats.nlargest(10, 'discrimination')
    extra_cols = ['task_type'] if 'task_type' in item_stats.columns else []
    print(top_disc[['accuracy', 'discrimination', 'category'] + extra_cols].to_string())

    # ── Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(item_stats['accuracy'], bins=20, color='#2196F3', alpha=0.8, edgecolor='white')
    ax.axvline(item_stats['accuracy'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={item_stats["accuracy"].mean():.2f}')
    ax.set_xlabel('Accuracy (fraction of models correct)', fontsize=11)
    ax.set_ylabel('Number of Items', fontsize=11)
    ax.set_title('Item Difficulty Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    colors_map = {'Easy (ceiling)': '#4CAF50', 'Hard (floor)': '#F44336',
                  'Highly Discriminative': '#FF9800', 'Medium': '#90CAF9'}
    for cat, color in colors_map.items():
        mask = item_stats['category'] == cat
        ax2.scatter(item_stats.loc[mask, 'difficulty'],
                    item_stats.loc[mask, 'discrimination'],
                    c=color, label=f'{cat} (n={mask.sum()})', alpha=0.7, s=40)
    ax2.set_xlabel('Difficulty (1 - accuracy)', fontsize=11)
    ax2.set_ylabel('Discrimination Index', fontsize=11)
    ax2.set_title('Item Discrimination vs Difficulty\n(IRT-lite)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    if 'task_type' in item_stats.columns:
        task_disc = item_stats.groupby('task_type')['discrimination'].agg(['mean', 'std']).reset_index()
        task_labels = [TASK_MAP.get(t, t[:3]) for t in task_disc['task_type']]
        ax3.bar(task_labels, task_disc['mean'],
                yerr=task_disc['std'], capsize=5,
                color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'], alpha=0.85, edgecolor='white')
        ax3.set_xlabel('Task Type', fontsize=11)
        ax3.set_ylabel('Mean Discrimination Index', fontsize=11)
        ax3.set_title('Item Discrimination by Task\n(higher = better benchmark quality)',
                      fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp7_item_discrimination.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    summary = {
        'highly_discriminative': int((item_stats['category'] == 'Highly Discriminative').sum()),
        'ceiling_items': int((item_stats['category'] == 'Easy (ceiling)').sum()),
        'floor_items': int((item_stats['category'] == 'Hard (floor)').sum()),
        'mean_discrimination': round(float(item_stats['discrimination'].mean()), 3),
        'median_accuracy': round(float(item_stats['accuracy'].median()), 3)
    }
    with open(OUTPUT / "discrimination_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("=== EXPERIMENT 7 COMPLETE ===")
    print(f"Highly discriminative items: {summary['highly_discriminative']}")
    print(f"Ceiling items (too easy): {summary['ceiling_items']}")
    print(f"Floor items (too hard): {summary['floor_items']}")


if __name__ == "__main__":
    main()
