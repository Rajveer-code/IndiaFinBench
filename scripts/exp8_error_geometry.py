"""
Experiment 8: Model Error Geometry + Dissociation Index
1. Pairwise model disagreement clustering
2. Dissociation Index: gap between CON and TMP performance
3. Model capability profile visualization
"""
import sys, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_correctness_matrix, get_task_accuracies, OUTPUT_DIR, FIGURES_DIR
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

OUTPUT = OUTPUT_DIR / "error_geometry"
OUTPUT.mkdir(parents=True, exist_ok=True)


def compute_dissociation_index(task_accs):
    """
    DI = (ACC_CON - ACC_TMP) / (ACC_CON + ACC_TMP)
    High DI → model can compare text pairs (CON) but fails temporal tracking (TMP)
    """
    di_results = []
    for model, row in task_accs.iterrows():
        con_acc = row.get('CON', np.nan)
        tmp_acc = row.get('TMP', np.nan)
        if pd.notna(con_acc) and pd.notna(tmp_acc) and (con_acc + tmp_acc) > 0:
            di = (con_acc - tmp_acc) / (con_acc + tmp_acc)
            di_results.append({'model': model, 'CON': con_acc, 'TMP': tmp_acc, 'DI': di})
    return pd.DataFrame(di_results).sort_values('DI', ascending=False)


def main():
    print("Loading data...")
    task_accs = get_task_accuracies()
    matrix = load_correctness_matrix()

    print("Computing Dissociation Index...")
    di_df = compute_dissociation_index(task_accs)
    di_df.to_csv(OUTPUT / "dissociation_index.csv", index=False)
    print("\nDissociation Index (CON vs TMP gap):")
    print(di_df[['model', 'CON', 'TMP', 'DI']].to_string(index=False))

    # ── Error similarity (Jaccard) ────────────────────────────────────────
    model_cols = matrix.columns.tolist()
    error_matrix = 1 - matrix.fillna(0)

    jaccard_data = np.zeros((len(model_cols), len(model_cols)))
    for i, m1 in enumerate(model_cols):
        for j, m2 in enumerate(model_cols):
            if i == j:
                jaccard_data[i][j] = 1.0
            else:
                both_error = ((error_matrix[m1] == 1) & (error_matrix[m2] == 1)).sum()
                either_error = ((error_matrix[m1] == 1) | (error_matrix[m2] == 1)).sum()
                jaccard_data[i][j] = both_error / either_error if either_error > 0 else 0

    jaccard_df = pd.DataFrame(jaccard_data, index=model_cols, columns=model_cols)
    jaccard_df.to_csv(OUTPUT / "error_jaccard_similarity.csv")

    # ── Figures ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    if len(di_df) > 0:
        colors = ['#F44336' if d > 0.1 else '#FF9800' if d > 0.05 else '#4CAF50'
                  for d in di_df['DI']]
        bars = ax1.barh(di_df['model'], di_df['DI'], color=colors, alpha=0.85)
        ax1.axvline(x=0, color='black', linewidth=0.8)
        ax1.set_xlabel('Dissociation Index (DI)', fontsize=11)
        ax1.set_title('CON–TMP Dissociation Index\n(High = good at comparison, poor at state tracking)',
                      fontsize=11, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        for bar, val in zip(bars, di_df['DI']):
            ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    if len(di_df) > 0:
        ax2.scatter(di_df['CON'], di_df['TMP'], s=100, c=di_df['DI'], cmap='RdYlGn_r',
                    alpha=0.8, edgecolors='black', linewidths=0.5)
        ax2.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, alpha=0.4, label='DI=0 line')
        for _, row in di_df.iterrows():
            ax2.annotate(str(row['model']).split()[0], (row['CON'], row['TMP']),
                         textcoords='offset points', xytext=(5, 5), fontsize=7)
        ax2.set_xlabel('CON Accuracy', fontsize=11)
        ax2.set_ylabel('TMP Accuracy', fontsize=11)
        ax2.set_title('CON vs TMP Performance\n(Points below diagonal = high dissociation)',
                      fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    short_names = [m.split()[0] for m in model_cols]
    im = ax3.imshow(jaccard_df.values, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax3.set_xticks(range(len(short_names)))
    ax3.set_yticks(range(len(short_names)))
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(short_names, fontsize=8)
    ax3.set_title('Error Similarity Matrix\n(Jaccard similarity of errors)',
                  fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    try:
        distance_matrix = 1 - jaccard_df.values
        np.fill_diagonal(distance_matrix, 0)
        condensed = pdist(distance_matrix)
        Z = linkage(condensed, method='ward')
        dendrogram(Z, labels=short_names, ax=ax4, leaf_rotation=45, leaf_font_size=9,
                   color_threshold=0.7 * max(Z[:, 2]))
        ax4.set_title('Model Clustering by Error Similarity\n(similar errors = same cluster)',
                      fontsize=11, fontweight='bold')
        ax4.set_ylabel('Error Distance', fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
    except Exception as e:
        print(f"  Dendrogram error: {e}")

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp8_error_geometry.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    if len(di_df) > 0:
        summary = {
            'max_dissociation_model': di_df.iloc[0]['model'],
            'max_DI': round(float(di_df.iloc[0]['DI']), 4),
            'min_dissociation_model': di_df.iloc[-1]['model'],
            'min_DI': round(float(di_df.iloc[-1]['DI']), 4),
            'mean_DI': round(float(di_df['DI'].mean()), 4)
        }
        with open(OUTPUT / "dissociation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    print("=== EXPERIMENT 8 COMPLETE ===")
    if len(di_df) > 0:
        print(f"Highest dissociation: {di_df.iloc[0]['model']} (DI={di_df.iloc[0]['DI']:.3f})")


if __name__ == "__main__":
    main()
