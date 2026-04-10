"""
generate_difficulty_plot.py — IndiaFinBench v7 (9 models)
Run: python generate_difficulty_plot.py
Output: figures/difficulty_lineplot.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

models = [
    'Claude 3 Haiku', 'Gemini 2.5 Flash', 'Llama 4 Scout 17B',
    'Qwen3-32B', 'LLaMA-3.3-70B', 'LLaMA-3-8B',
    'Gemma 4 E4B', 'Mistral-7B', 'DeepSeek R1 70B'
]
accuracy_by_difficulty = {
    'Claude 3 Haiku':    [90.8, 92.3, 90.0],
    'Gemini 2.5 Flash':  [95.4, 84.6, 70.0],
    'Llama 4 Scout 17B': [86.2, 76.9, 90.0],
    'Qwen3-32B':         [81.5, 83.1, 85.0],
    'LLaMA-3.3-70B':     [81.5, 78.5, 90.0],
    'LLaMA-3-8B':        [81.5, 69.2, 75.0],
    'Gemma 4 E4B':       [90.8, 64.6, 55.0],
    'Mistral-7B':        [72.3, 70.8, 80.0],
    'DeepSeek R1 70B':   [72.3, 69.2, 70.0],
}
COLORS  = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#FF5722','#795548','#607D8B']
MARKERS = ['o','s','^','D','v','P','h','*','X']

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor('#f9f9f9'); fig.patch.set_facecolor('white')
x = np.array([0, 1, 2])

for i, (model, col, mk) in enumerate(zip(models, COLORS, MARKERS)):
    lw = 2.5 if i < 2 else (2.0 if i < 5 else 1.5)
    ls = '-'  if i < 5 else '--'
    ax.plot(x, accuracy_by_difficulty[model], marker=mk, color=col,
            linewidth=lw, linestyle=ls, markersize=8, label=model, alpha=0.88)

ax.axhline(60, color='#999', linestyle=':', linewidth=1.2, label='Human Expert (60.0%)')
ax.set_xticks(x)
ax.set_xticklabels(['Easy\n(n=65)', 'Medium\n(n=65)', 'Hard\n(n=20)'], fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(45, 102)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.5, color='grey')
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower left', fontsize=8.5, ncol=2, framealpha=0.92, edgecolor='#cccccc')
ax.set_title('Model Accuracy by Difficulty Level (9 Models)', fontsize=13, fontweight='bold', pad=12)

import os; os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/difficulty_lineplot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved figures/difficulty_lineplot.png")
