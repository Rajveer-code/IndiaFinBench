"""
generate_heatmap.py — IndiaFinBench v7 (9 models)
Run: python generate_heatmap.py
Output: figures/inter_task_correlation.png
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

model_accuracies = {
    'REG': [92.5, 96.2, 79.2, 77.4, 77.4, 77.4, 90.6, 69.8, 60.4],
    'NUM': [93.8, 84.4, 75.0, 75.0, 84.4, 62.5, 65.6, 68.8, 78.1],
    'CON': [86.7, 83.3,100.0, 86.7, 90.0, 86.7, 76.7, 80.0, 93.3],
    'TMP': [91.4, 80.0, 80.0, 94.3, 77.1, 74.3, 57.1, 74.3, 60.0],
}
tasks = ['REG', 'NUM', 'CON', 'TMP']
corr_matrix = np.zeros((4, 4))
for i, t1 in enumerate(tasks):
    for j, t2 in enumerate(tasks):
        r, _ = spearmanr(model_accuracies[t1], model_accuracies[t2])
        corr_matrix[i, j] = r

fig, ax = plt.subplots(figsize=(6, 5.5))
fig.patch.set_facecolor('white')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', xticklabels=tasks, yticklabels=tasks,
            cmap='RdYlGn', vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Spearman ρ', 'shrink': 0.85},
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})
ax.set_xticklabels(tasks, fontsize=12, fontweight='bold')
ax.set_yticklabels(tasks, fontsize=12, fontweight='bold', rotation=0)
ax.set_title('Inter-Task Spearman Correlation\n(Model Accuracy Vectors, n=9 models)',
             fontsize=11, fontweight='bold', pad=10)

import os; os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/inter_task_correlation.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved figures/inter_task_correlation.png")
