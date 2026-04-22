import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Per-model accuracy vectors across 4 tasks (12 models)
REG = [93.1, 85.1, 86.2, 86.2, 89.1, 79.9, 79.9, 79.9, 89.7, 79.9, 72.4, 83.9]
NUM = [84.8, 77.2, 75.0, 66.3, 65.2, 64.1, 59.8, 58.7, 48.9, 66.3, 69.6, 50.0]
CON = [88.7, 90.3, 95.2, 98.4, 91.9, 93.5, 95.2, 95.2, 93.5, 80.6, 96.8, 72.6]
TMP = [88.5, 92.3, 79.5, 84.6, 75.6, 78.2, 76.9, 76.9, 64.1, 74.4, 70.5, 62.8]

tasks = ['REG', 'NUM', 'CON', 'TMP']
vectors = [REG, NUM, CON, TMP]

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

plt.title('Inter-Task Correlation\n(Spearman ρ, n=12 models)',
          fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/figure4_correlation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/figure4_correlation.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved.")