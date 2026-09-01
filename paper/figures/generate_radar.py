import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# Loaded live from the canonical task-accuracy matrix so this figure cannot
# drift from the numbers in the paper. Previously hardcoded and went stale
# after the Gemma re-run (CON/TMP columns changed materially).
CSV_PATH = pathlib.Path(__file__).resolve().parents[2] / "evaluation" / "task_accuracy_matrix.csv"
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = {r["model"]: r for r in csv.DictReader(f)}

DISPLAY = {'DeepSeek R1 70B': 'DeepSeek-R1-Distill-Llama-70B'}
categories = ['REG', 'NUM', 'CON', 'TMP']
scores = {m: [float(rows[m][c]) for c in categories] for m in rows}

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Highlight top-3 + bottom-1; rest in grey
highlight = {
    'Gemini 2.5 Flash': ('#2166ac', 2.5, '-'),
    'Qwen3-32B':        ('#d6604d', 2.0, '-'),
    'LLaMA-3.3-70B':    ('#4dac26', 1.8, '-'),
    'Gemma 3 4B':       ('#7b3294', 1.5, '--'),
}

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for model, vals in scores.items():
    vals_plot = vals + vals[:1]
    if model in highlight:
        color, lw, ls = highlight[model]
        ax.plot(angles, vals_plot, color=color, linewidth=lw,
                linestyle=ls, label=DISPLAY.get(model, model))
        ax.fill(angles, vals_plot, color=color, alpha=0.07)
    else:
        ax.plot(angles, vals_plot, color='#aaaaaa', linewidth=0.8,
                linestyle='-', alpha=0.5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Regulatory\nInterpretation', 'Numerical\nReasoning',
                    'Contradiction\nDetection', 'Temporal\nReasoning'],
                   fontsize=10, fontweight='bold')
ax.set_ylim(40, 100)
ax.set_yticks([50, 60, 70, 80, 90, 100])
ax.set_yticklabels(['50', '60', '70', '80', '90', '100'], fontsize=7, color='grey')
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8.5)
plt.title('IndiaFinBench — Per-Task Accuracy Profiles', fontsize=11,
          fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('paper/figures/figure2_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/figure2_radar.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved.")
