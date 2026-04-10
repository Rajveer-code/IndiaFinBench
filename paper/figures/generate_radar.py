import numpy as np
import matplotlib.pyplot as plt

models = ['Claude 3 Haiku', 'Gemini 2.5 Flash', 'LLaMA-3.3-70B', 'LLaMA-3-8B', 'Mistral-7B']
scores = {
    'Claude 3 Haiku':   [92.5, 93.8, 86.7, 91.4],
    'Gemini 2.5 Flash': [96.2, 84.4, 83.3, 82.4],
    'LLaMA-3.3-70B':    [77.4, 84.4, 90.0, 77.1],
    'LLaMA-3-8B':       [77.4, 62.5, 86.7, 74.3],
    'Mistral-7B':       [69.8, 68.8, 80.0, 74.3],
}
labels = ['REG', 'NUM', 'CON', 'TMP']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
N = 4
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for idx, (model, vals) in enumerate(scores.items()):
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, color=colors[idx], linewidth=2, label=model)
    ax.fill(angles, vals_plot, color=colors[idx], alpha=0.08)

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)
ax.set_ylim(60, 100)
ax.set_rgrids([70, 80, 90, 100], labels=['70%', '80%', '90%', '100%'], fontsize=9)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=10)
ax.set_title('Model Capability Profiles Across Task Types', y=1.10, fontsize=14)
plt.tight_layout()
plt.savefig('figures/radar_chart.png', dpi=300, bbox_inches='tight')
print("Saved figures/radar_chart.png")
