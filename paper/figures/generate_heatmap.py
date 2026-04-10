import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

model_accuracies = {
    'REG': [92.5, 96.2, 77.4, 77.4, 69.8],
    'NUM': [93.8, 84.4, 84.4, 62.5, 68.8],
    'CON': [86.7, 83.3, 90.0, 86.7, 80.0],
    'TMP': [91.4, 82.4, 77.1, 74.3, 74.3],
}
tasks = ['REG', 'NUM', 'CON', 'TMP']
corr_matrix = np.zeros((4, 4))
for i, t1 in enumerate(tasks):
    for j, t2 in enumerate(tasks):
        r, _ = spearmanr(model_accuracies[t1], model_accuracies[t2])
        corr_matrix[i, j] = r

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', xticklabels=tasks,
            yticklabels=tasks, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax,
            cbar_kws={'label': 'Spearman \u03c1'})
ax.set_title("Inter-Task Spearman Correlation (Model Accuracy Vectors)")
plt.tight_layout()
plt.savefig('figures/inter_task_correlation.png', dpi=300)
print("Saved figures/inter_task_correlation.png")
