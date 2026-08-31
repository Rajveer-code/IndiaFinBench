import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

models = [
    'Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B', 'Llama 4 Scout 17B',
    'Kimi K2', 'LLaMA-3-8B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'Gemini 2.5 Pro', 'Mistral-7B', 'DeepSeek R1 70B', 'Gemma 4 E4B'
]

# Data is loaded from the evaluation results files, never hardcoded, so this
# figure cannot drift from the numbers in the paper. Verified 2026-08-31:
# the previous hardcoded block matched the data on all 48 cells.
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from scripts.novel_methods_utils import get_task_accuracies

_acc = (get_task_accuracies() * 100).round(1)
data = {t: [float(_acc.loc[m, t]) for m in models] for t in ['REG', 'NUM', 'CON', 'TMP']}

tasks = ['REG', 'NUM', 'CON', 'TMP']
matrix = np.array([[data[t][i] for t in tasks] for i in range(len(models))])

fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(matrix, cmap='YlGn', vmin=45, vmax=100, aspect='auto')

ax.set_xticks(range(len(tasks)))
ax.set_xticklabels(['Regulatory\nInterpretation', 'Numerical\nReasoning',
                    'Contradiction\nDetection', 'Temporal\nReasoning'],
                   fontsize=10, fontweight='bold')
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

for i in range(len(models)):
    for j in range(len(tasks)):
        val = matrix[i, j]
        color = 'white' if val > 85 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=8.5, color=color, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Accuracy (%)', fontsize=9)

plt.title('IndiaFinBench — Model × Task Accuracy Heatmap', fontsize=11,
          fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig('paper/figures/figure1_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/figure1_heatmap.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved.")