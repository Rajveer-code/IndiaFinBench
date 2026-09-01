import csv
import pathlib

import matplotlib.pyplot as plt

# Loaded live from the canonical difficulty breakdown so this figure cannot
# drift from the numbers in the paper. Previously hardcoded and went stale
# after the Gemma re-run (Medium/Hard rows changed materially).
CSV_PATH = pathlib.Path(__file__).resolve().parents[2] / "evaluation" / "difficulty_breakdown.csv"
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = {r["model"]: r for r in csv.DictReader(f)}

DISPLAY = {'DeepSeek R1 70B': 'DeepSeek-R1-Distill-Llama-70B'}
models = list(rows.keys())
accuracy_by_difficulty = {
    m: [float(rows[m]["easy_acc"]), float(rows[m]["medium_acc"]), float(rows[m]["hard_acc"])]
    for m in models
}

x = [0, 1, 2]
easy_n, med_n, hard_n = rows[models[0]]["easy_n"], rows[models[0]]["medium_n"], rows[models[0]]["hard_n"]
x_labels = [f'Easy\n(n={easy_n})', f'Medium\n(n={med_n})', f'Hard\n(n={hard_n})']

# Colour scheme: tier-based
tier1_color = '#2166ac'
tier2_color = '#aaaaaa'
tier3_color = '#d6604d'

tier1 = ['Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B',
         'Llama 4 Scout 17B', 'Kimi K2']
tier3 = ['Gemma 3 4B']

fig, ax = plt.subplots(figsize=(8, 6))

for model, vals in accuracy_by_difficulty.items():
    if model in tier1:
        color, lw, zorder = tier1_color, 2.0, 3
    elif model in tier3:
        color, lw, zorder = tier3_color, 2.0, 3
    else:
        color, lw, zorder = tier2_color, 1.0, 2

    ax.plot(x, vals, marker='o', color=color, linewidth=lw,
            markersize=5, zorder=zorder,
            label=DISPLAY.get(model, model) if (model in tier1 or model in tier3) else '_nolegend_',
            alpha=0.85 if color != tier2_color else 0.5)

# Label endpoints for key models
key_labels = ['Gemini 2.5 Flash', 'Gemma 3 4B', 'LLaMA-3.3-70B', 'Gemini 2.5 Pro']
for model in key_labels:
    vals = accuracy_by_difficulty[model]
    ax.annotate(DISPLAY.get(model, model), xy=(2, vals[2]), xytext=(2.05, vals[2]),
                fontsize=7.5, va='center')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=10)
ax.set_ylim(50, 100)
ax.set_xlim(-0.2, 2.8)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_title('IndiaFinBench — Accuracy by Difficulty Level', fontsize=11,
             fontweight='bold')
ax.legend(loc='lower left', fontsize=8, title='Tier 1 & Tier 3 models',
          title_fontsize=8)

plt.tight_layout()
plt.savefig('paper/figures/figure3_difficulty.pdf', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/figure3_difficulty.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved.")
