import matplotlib.pyplot as plt
import numpy as np

models = [
    'Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B', 'Llama 4 Scout 17B',
    'Kimi K2', 'LLaMA-3-8B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'Gemini 2.5 Pro', 'Mistral-7B', 'DeepSeek R1 70B', 'Gemma 4 E4B'
]

# Easy(n=160), Medium(n=182), Hard(n=64) — verified from 406-item CSVs
accuracy_by_difficulty = {
    'Gemini 2.5 Flash':  [92.5, 89.0, 84.4],
    'Qwen3-32B':         [81.9, 87.9, 87.5],
    'LLaMA-3.3-70B':     [79.4, 85.2, 90.6],
    'Llama 4 Scout 17B': [82.5, 81.9, 89.1],
    'Kimi K2':           [81.9, 80.8, 82.8],
    'LLaMA-3-8B':        [76.2, 79.7, 78.1],
    'GPT-OSS 120B':      [79.4, 76.4, 73.4],
    'GPT-OSS 20B':       [75.0, 79.7, 73.4],
    'Gemini 2.5 Pro':    [83.1, 72.5, 68.8],   # was blank row — now filled
    'Mistral-7B':        [74.4, 76.9, 76.6],
    'DeepSeek R1 70B':   [72.5, 77.5, 75.0],
    'Gemma 4 E4B':       [82.5, 64.8, 56.2],
}

x = [0, 1, 2]
x_labels = ['Easy\n(n=160)', 'Medium\n(n=182)', 'Hard\n(n=64)']

# Colour scheme: tier-based
tier1_color = '#2166ac'
tier2_color = '#aaaaaa'
tier3_color = '#d6604d'

tier1 = ['Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B',
         'Llama 4 Scout 17B', 'Kimi K2']
tier3 = ['Gemma 4 E4B']

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
            label=model if (model in tier1 or model in tier3) else '_nolegend_',
            alpha=0.85 if color != tier2_color else 0.5)

# Label endpoints for key models
key_labels = ['Gemini 2.5 Flash', 'Gemma 4 E4B', 'LLaMA-3.3-70B', 'Gemini 2.5 Pro']
for model in key_labels:
    vals = accuracy_by_difficulty[model]
    ax.annotate(model, xy=(2, vals[2]), xytext=(2.05, vals[2]),
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