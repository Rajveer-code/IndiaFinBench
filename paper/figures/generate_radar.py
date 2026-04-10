"""
generate_radar.py — IndiaFinBench v7 (9 models)
Run: python generate_radar.py
Output: figures/radar_chart.png
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
scores = {
    'Claude 3 Haiku':    [92.5, 93.8, 86.7, 91.4],
    'Gemini 2.5 Flash':  [96.2, 84.4, 83.3, 80.0],
    'Llama 4 Scout 17B': [79.2, 75.0,100.0, 80.0],
    'Qwen3-32B':         [77.4, 75.0, 86.7, 94.3],
    'LLaMA-3.3-70B':     [77.4, 84.4, 90.0, 77.1],
    'LLaMA-3-8B':        [77.4, 62.5, 86.7, 74.3],
    'Gemma 4 E4B':       [90.6, 65.6, 76.7, 57.1],
    'Mistral-7B':        [69.8, 68.8, 80.0, 74.3],
    'DeepSeek R1 70B':   [60.4, 78.1, 93.3, 60.0],
}
COLORS  = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336','#00BCD4','#FF5722','#795548','#607D8B']
MARKERS = ['o','s','^','D','v','P','h','*','X']
labels  = ['REG\n(Reg. Interp.)','NUM\n(Num. Reasoning)','CON\n(Contradiction\nDet.)','TMP\n(Temporal\nReasoning)']

N = 4
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')

for i, (model, col, mk) in enumerate(zip(models, COLORS, MARKERS)):
    vals = scores[model] + [scores[model][0]]
    lw = 2.5 if i < 2 else (2.0 if i < 5 else 1.5)
    ls = '-'  if i < 5 else '--'
    ax.plot(angles, vals, marker=mk, color=col, linewidth=lw,
            linestyle=ls, markersize=5, label=model, alpha=0.88)
    ax.fill(angles, vals, alpha=0.05, color=col)

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
ax.set_ylim(50, 105)
ax.set_rgrids([60,70,80,90,100], labels=['60%','70%','80%','90%','100%'], fontsize=9, color='grey')
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
ax.spines['polar'].set_visible(False)
ax.legend(models, loc='upper right', bbox_to_anchor=(1.45,1.18),
          fontsize=9, framealpha=0.92, edgecolor='#cccccc')
ax.set_title('Model Capability Profiles Across Task Types', y=1.12, fontsize=14, fontweight='bold')

import os; os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved figures/radar_chart.png")
