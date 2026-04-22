import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

models = [
    'Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B', 'Llama 4 Scout 17B',
    'Kimi K2', 'LLaMA-3-8B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'Gemini 2.5 Pro', 'Mistral-7B', 'DeepSeek R1 70B', 'Gemma 4 E4B'
]

data = {
    'REG': [93.1, 85.1, 86.2, 86.2, 89.1, 79.9, 79.9, 79.9, 89.7, 79.9, 72.4, 83.9],
    'NUM': [84.8, 77.2, 75.0, 66.3, 65.2, 64.1, 59.8, 58.7, 48.9, 66.3, 69.6, 50.0],
    'CON': [88.7, 90.3, 95.2, 98.4, 91.9, 93.5, 95.2, 95.2, 93.5, 80.6, 96.8, 72.6],
    'TMP': [88.5, 92.3, 79.5, 84.6, 75.6, 78.2, 76.9, 76.9, 64.1, 74.4, 70.5, 62.8],
}

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