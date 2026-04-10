import matplotlib.pyplot as plt
import numpy as np

models = ['Claude 3 Haiku', 'Gemini 2.5 Flash', 'LLaMA-3.3-70B', 'LLaMA-3-8B', 'Mistral-7B']
difficulties = ['Easy', 'Medium', 'Hard']
accuracy_by_difficulty = {
    'Claude 3 Haiku':   [90.8, 92.3, 90.0],
    'Gemini 2.5 Flash': [95.4, 84.6, 73.7],
    'LLaMA-3.3-70B':    [81.5, 78.5, 90.0],
    'LLaMA-3-8B':       [81.5, 69.2, 75.0],
    'Mistral-7B':       [72.3, 70.8, 80.0],
}
colors  = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
markers = ['o', 's', '^', 'D', 'v']

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(difficulties))

for idx, model in enumerate(models):
    ax.plot(x, accuracy_by_difficulty[model], marker=markers[idx],
            color=colors[idx], linewidth=2, markersize=8, label=model)

ax.set_xticks(x)
ax.set_xticklabels(difficulties, fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(60, 100)
ax.set_title('Model Accuracy by Difficulty Level', fontsize=13)
ax.legend(fontsize=9, loc='lower left')
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('figures/difficulty_lineplot.png', dpi=300)
print("Saved figures/difficulty_lineplot.png")
