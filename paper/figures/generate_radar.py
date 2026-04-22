import matplotlib.pyplot as plt
import numpy as np

models = [
    'Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B', 'Llama 4 Scout 17B',
    'Kimi K2', 'LLaMA-3-8B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'Gemini 2.5 Pro', 'Mistral-7B', 'DeepSeek R1 70B', 'Gemma 4 E4B'
]

scores = {
    'Gemini 2.5 Flash':  [93.1, 84.8, 88.7, 88.5],
    'Qwen3-32B':         [85.1, 77.2, 90.3, 92.3],
    'LLaMA-3.3-70B':     [86.2, 75.0, 95.2, 79.5],
    'Llama 4 Scout 17B': [86.2, 66.3, 98.4, 84.6],
    'Kimi K2':           [89.1, 65.2, 91.9, 75.6],
    'LLaMA-3-8B':        [79.9, 64.1, 93.5, 78.2],
    'GPT-OSS 120B':      [79.9, 59.8, 95.2, 76.9],
    'GPT-OSS 20B':       [79.9, 58.7, 95.2, 76.9],
    'Gemini 2.5 Pro':    [89.7, 48.9, 93.5, 64.1],
    'Mistral-7B':        [79.9, 66.3, 80.6, 74.4],
    'DeepSeek R1 70B':   [72.4, 69.6, 96.8, 70.5],
    'Gemma 4 E4B':       [83.9, 50.0, 72.6, 62.8],
}

categories = ['REG', 'NUM', 'CON', 'TMP']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Highlight top-3 + bottom-1; rest in grey
highlight = {
    'Gemini 2.5 Flash': ('#2166ac', 2.5, '-'),
    'Qwen3-32B':        ('#d6604d', 2.0, '-'),
    'LLaMA-3.3-70B':    ('#4dac26', 1.8, '-'),
    'Gemma 4 E4B':      ('#7b3294', 1.5, '--'),
}

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for model, vals in scores.items():
    vals_plot = vals + vals[:1]
    if model in highlight:
        color, lw, ls = highlight[model]
        ax.plot(angles, vals_plot, color=color, linewidth=lw,
                linestyle=ls, label=model)
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