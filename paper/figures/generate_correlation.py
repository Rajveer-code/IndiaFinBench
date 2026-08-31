import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Per-model accuracy vectors across 4 tasks (12 models), loaded live from the
# canonical task-accuracy matrix so this figure cannot drift from the numbers
# in the paper. Previously hardcoded and went stale after the Gemma re-run.
CSV_PATH = pathlib.Path(__file__).resolve().parents[2] / "evaluation" / "task_accuracy_matrix.csv"
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = {r["model"]: r for r in csv.DictReader(f)}

tasks = ['REG', 'NUM', 'CON', 'TMP']
vectors = [[float(rows[m][t]) for m in rows] for t in tasks]

corr_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        rho, _ = spearmanr(vectors[i], vectors[j])
        corr_matrix[i, j] = rho

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(tasks, fontsize=11, fontweight='bold')
ax.set_yticklabels(tasks, fontsize=11, fontweight='bold')

for i in range(4):
    for j in range(4):
        val = corr_matrix[i, j]
        color = 'white' if abs(val) > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=11, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Spearman ρ", fontsize=9)

plt.title(f'Inter-Task Correlation\n(Spearman ρ, n={len(rows)} models)',
          fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/figure4_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/figure4_correlation.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved.")
