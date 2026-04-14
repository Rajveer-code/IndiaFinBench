"""
generate_figures.py — IndiaFinBench v11
=======================================
Single reproducible script that:
  1. Loads all 406-item evaluation CSVs
  2. Computes per-difficulty accuracy breakdowns
  3. Runs pairwise bootstrap significance tests (10,000 resamples)
  4. Computes 95% Wilson score confidence intervals
  5. Generates all four paper figures + saves JSON/CSV outputs

Run from the project root:
    python scripts/generate_figures.py

Outputs:
    paper/figures/performance_heatmap.png
    paper/figures/radar_chart.png
    paper/figures/difficulty_lineplot.png
    paper/figures/inter_task_correlation.png
    evaluation/bootstrap_significance_results.json
    evaluation/wilson_ci_results.json
    evaluation/difficulty_breakdown.csv
    evaluation/task_accuracy_matrix.csv
"""

import os
import csv
import json
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr

# ── Reproducibility ────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'evaluation', 'results')
EVAL_OUT    = os.path.join(ROOT, 'evaluation')
FIG_OUT     = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG_OUT,  exist_ok=True)
os.makedirs(EVAL_OUT, exist_ok=True)

# ── Model registry (406-item evaluations) ─────────────────────────────────
# Ordered by overall accuracy (highest first)
MODEL_FILES = {
    'Gemini 2.5 Flash':    'gemini_results.csv',
    'Qwen3-32B':           'qwen3_32b_results.csv',
    'LLaMA-3.3-70B':       'groq70b_results.csv',
    'Llama 4 Scout 17B':   'llama4scout_results.csv',
    'Kimi K2':             'kimi_k2_results.csv',
    'LLaMA-3-8B':          'llama3_results.csv',
    'GPT-OSS 120B':        'gpt_oss_120b_results.csv',
    'GPT-OSS 20B':         'gpt_oss_20b_results.csv',
    'Mistral-7B':          'mistral_results.csv',
    'DeepSeek R1 70B':     'deepseek_r1_70b_results.csv',
    'Gemma 4 E4B':         'gemma4_e4b_results.csv',
}

# Claude 3 Haiku evaluated on the initial 150-item subset (kept separate)
HAIKU_STATS = {
    'overall': 91.3, 'REG': 92.5, 'NUM': 93.8, 'CON': 86.7, 'TMP': 91.4,
    'n': 150, 'n_reg': 53, 'n_num': 32, 'n_con': 30, 'n_tmp': 35,
}

TASK_MAP = {
    'regulatory_interpretation': 'REG',
    'numerical_reasoning':       'NUM',
    'contradiction_detection':   'CON',
    'temporal_reasoning':        'TMP',
}

# ── Style constants ────────────────────────────────────────────────────────
COLORS = [
    '#1565C0',  # Gemini 2.5 Flash     — deep blue
    '#6A1B9A',  # Qwen3-32B            — deep purple
    '#2E7D32',  # LLaMA-3.3-70B        — dark green
    '#E65100',  # Llama 4 Scout 17B    — deep orange
    '#AD1457',  # Kimi K2              — deep pink
    '#00695C',  # LLaMA-3-8B           — teal
    '#4527A0',  # GPT-OSS 120B         — indigo
    '#283593',  # GPT-OSS 20B          — navy
    '#4E342E',  # Mistral-7B           — brown
    '#37474F',  # DeepSeek R1 70B      — blue-grey
    '#BF360C',  # Gemma 4 E4B          — deep red-orange
]
MARKERS = ['o', 's', '^', 'D', 'P', 'v', 'h', 'X', '*', '<', '>']

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_csv(fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_model_data():
    """Return dict: model_name → {rows, per_task_correct, per_task_total,
                                   overall_acc, per_task_acc, difficulty_acc}"""
    data = {}
    for model, fname in MODEL_FILES.items():
        rows = load_csv(fname)
        assert len(rows) == 406, f"{fname}: expected 406 rows, got {len(rows)}"

        per_task_correct = {t: 0 for t in TASK_MAP.values()}
        per_task_total   = {t: 0 for t in TASK_MAP.values()}
        diff_correct     = {'easy': 0, 'medium': 0, 'hard': 0}
        diff_total       = {'easy': 0, 'medium': 0, 'hard': 0}

        for r in rows:
            tt   = TASK_MAP[r['task_type']]
            d    = r['difficulty'].lower()
            corr = int(r['correct'])
            per_task_correct[tt] += corr
            per_task_total[tt]   += 1
            diff_correct[d]      += corr
            diff_total[d]        += 1

        per_task_acc = {
            t: per_task_correct[t] / per_task_total[t] * 100
            for t in per_task_correct
        }
        overall_acc  = sum(per_task_correct.values()) / 406 * 100
        diff_acc     = {
            d: diff_correct[d] / diff_total[d] * 100
            for d in diff_correct
        }

        data[model] = {
            'rows':            rows,
            'per_task_correct': per_task_correct,
            'per_task_total':   per_task_total,
            'overall_acc':      overall_acc,
            'per_task_acc':     per_task_acc,
            'diff_acc':         diff_acc,
            'diff_total':       diff_total,
        }
    return data


print("Loading evaluation CSVs…")
MODEL_DATA = build_model_data()
MODELS     = list(MODEL_DATA.keys())    # in MODEL_FILES insertion order

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Statistical computations
# ══════════════════════════════════════════════════════════════════════════

# ── 2a. Bootstrap significance test ───────────────────────────────────────

def bootstrap_pvalue(correct_a, correct_b, n_resamples=10_000):
    """
    Paired bootstrap test.
    correct_a, correct_b : arrays of 0/1 per item (same items, same order).
    Returns two-tailed p-value (H0: acc_A == acc_B).
    """
    diff_a = np.array(correct_a) - np.array(correct_b)
    obs    = np.mean(diff_a)
    # Centre the differences so the null mean is 0
    centred = diff_a - np.mean(diff_a)
    n       = len(centred)
    count   = 0
    for _ in range(n_resamples):
        sample = centred[np.random.randint(0, n, n)]
        if abs(np.mean(sample)) >= abs(obs):
            count += 1
    return count / n_resamples


def get_correct_vector(model_name):
    return [int(r['correct']) for r in MODEL_DATA[model_name]['rows']]


print("Running bootstrap significance tests (10,000 resamples)…")
bootstrap_results = {}
for i, ma in enumerate(MODELS):
    for mb in MODELS[i+1:]:
        va = get_correct_vector(ma)
        vb = get_correct_vector(mb)
        p  = bootstrap_pvalue(va, vb)
        key = f"{ma} vs {mb}"
        bootstrap_results[key] = {
            'model_a':    ma,
            'model_b':    mb,
            'acc_a':      round(MODEL_DATA[ma]['overall_acc'], 2),
            'acc_b':      round(MODEL_DATA[mb]['overall_acc'], 2),
            'delta':      round(MODEL_DATA[ma]['overall_acc'] - MODEL_DATA[mb]['overall_acc'], 2),
            'p_value':    round(p, 4),
            'significant': p < 0.05,
        }

out_path = os.path.join(EVAL_OUT, 'bootstrap_significance_results.json')
with open(out_path, 'w') as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"  Saved {out_path}")

# Summary: significant pairs
sig_pairs = [(k, v) for k, v in bootstrap_results.items() if v['significant']]
print(f"  Significant pairs (p<0.05): {len(sig_pairs)} / {len(bootstrap_results)}")

# ── 2b. Wilson confidence intervals ───────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """95% Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    centre = (p_hat + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / (1 + z**2/n)
    return (max(0.0, (centre - margin) * 100), min(100.0, (centre + margin) * 100))


wilson_results = {}
for model in MODELS:
    d = MODEL_DATA[model]
    entry = {}
    # Overall
    total_correct = sum(d['per_task_correct'].values())
    lo, hi = wilson_ci(total_correct, 406)
    entry['overall'] = {'acc': round(d['overall_acc'], 2), 'ci_lo': round(lo,2), 'ci_hi': round(hi,2), 'n': 406}
    # Per task
    for t in ['REG','NUM','CON','TMP']:
        k = d['per_task_correct'][t]
        n = d['per_task_total'][t]
        lo, hi = wilson_ci(k, n)
        entry[t] = {'acc': round(d['per_task_acc'][t],2), 'ci_lo': round(lo,2), 'ci_hi': round(hi,2), 'n': n}
    wilson_results[model] = entry

# Haiku (150-item subset)
hk = HAIKU_STATS
wilson_results['Claude 3 Haiku†'] = {
    'overall': {'acc': hk['overall'], 'ci_lo': round(wilson_ci(137, 150)[0],2),
                'ci_hi': round(wilson_ci(137, 150)[1],2), 'n': 150},
    'REG': {'acc': hk['REG'], 'ci_lo': round(wilson_ci(49, 53)[0],2),
            'ci_hi': round(wilson_ci(49, 53)[1],2), 'n': 53},
    'NUM': {'acc': hk['NUM'], 'ci_lo': round(wilson_ci(30, 32)[0],2),
            'ci_hi': round(wilson_ci(30, 32)[1],2), 'n': 32},
    'CON': {'acc': hk['CON'], 'ci_lo': round(wilson_ci(26, 30)[0],2),
            'ci_hi': round(wilson_ci(26, 30)[1],2), 'n': 30},
    'TMP': {'acc': hk['TMP'], 'ci_lo': round(wilson_ci(32, 35)[0],2),
            'ci_hi': round(wilson_ci(32, 35)[1],2), 'n': 35},
}

out_path = os.path.join(EVAL_OUT, 'wilson_ci_results.json')
with open(out_path, 'w') as f:
    json.dump(wilson_results, f, indent=2)
print(f"  Saved {out_path}")

# ── 2c. Difficulty breakdown CSV ──────────────────────────────────────────

rows_out = []
for model in MODELS:
    d = MODEL_DATA[model]
    row = {'model': model}
    for diff in ['easy', 'medium', 'hard']:
        acc = d['diff_acc'].get(diff, 0.0)
        row[f'{diff}_acc']   = round(acc, 1)
        row[f'{diff}_n']     = d['diff_total'].get(diff, 0)
    rows_out.append(row)

out_path = os.path.join(EVAL_OUT, 'difficulty_breakdown.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','easy_acc','easy_n','medium_acc','medium_n','hard_acc','hard_n'])
    w.writeheader()
    w.writerows(rows_out)
print(f"  Saved {out_path}")

# Print difficulty table for paper
print("\nDifficulty breakdown (406-item models):")
print(f"{'Model':<22} {'Easy':>8} {'Medium':>9} {'Hard':>8}")
for row in rows_out:
    print(f"  {row['model']:<22} {row['easy_acc']:>6.1f}%  {row['medium_acc']:>7.1f}%  {row['hard_acc']:>6.1f}%")

# ── 2d. Task accuracy matrix ──────────────────────────────────────────────

matrix_rows = []
for model in MODELS:
    d = MODEL_DATA[model]
    matrix_rows.append({
        'model':   model,
        'REG':     round(d['per_task_acc']['REG'], 1),
        'NUM':     round(d['per_task_acc']['NUM'], 1),
        'CON':     round(d['per_task_acc']['CON'], 1),
        'TMP':     round(d['per_task_acc']['TMP'], 1),
        'Overall': round(d['overall_acc'], 1),
    })

out_path = os.path.join(EVAL_OUT, 'task_accuracy_matrix.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','REG','NUM','CON','TMP','Overall'])
    w.writeheader()
    w.writerows(matrix_rows)
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Figures
# ══════════════════════════════════════════════════════════════════════════

# Shorthand scores for figures
SCORES = {m: MODEL_DATA[m]['per_task_acc'] for m in MODELS}
OVERALL = {m: round(MODEL_DATA[m]['overall_acc'], 1) for m in MODELS}
DIFF_ACC = {m: MODEL_DATA[m]['diff_acc'] for m in MODELS}

# ── Figure 1: Performance Heatmap (model × task) ──────────────────────────

print("\nGenerating Figure 1: Performance Heatmap…")

tasks_labels = ['REG\n(174 items)', 'NUM\n(92 items)', 'CON\n(62 items)', 'TMP\n(78 items)', 'Overall\n(406 items)']

matrix = np.zeros((len(MODELS), 5))
for i, model in enumerate(MODELS):
    matrix[i, 0] = SCORES[model]['REG']
    matrix[i, 1] = SCORES[model]['NUM']
    matrix[i, 2] = SCORES[model]['CON']
    matrix[i, 3] = SCORES[model]['TMP']
    matrix[i, 4] = OVERALL[model]

# Shorten model names for y-axis
short_names = [
    'Gemini 2.5 Flash', 'Qwen3-32B', 'LLaMA-3.3-70B', 'Llama 4 Scout',
    'Kimi K2', 'LLaMA-3-8B', 'GPT-OSS 120B', 'GPT-OSS 20B',
    'Mistral-7B', 'DeepSeek R1', 'Gemma 4 E4B',
]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

im = ax.imshow(matrix, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')

# Annotate cells
for i in range(len(MODELS)):
    for j in range(5):
        val = matrix[i, j]
        color = 'white' if val < 68 or val > 93 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color)

ax.set_xticks(range(5))
ax.set_xticklabels(tasks_labels, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(short_names, fontsize=10)

# Add a vertical separator before the Overall column
ax.axvline(3.5, color='#555', linewidth=1.5, linestyle='--', alpha=0.6)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Accuracy (%)', fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    'IndiaFinBench: Model Accuracy by Task Type (Full 406-Item Evaluation)',
    fontsize=12, fontweight='bold', pad=14,
)
plt.tight_layout()
out = os.path.join(FIG_OUT, 'performance_heatmap.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out}")

# ── Figure 2: Radar Chart ─────────────────────────────────────────────────

print("Generating Figure 2: Radar Chart…")

# Use 11 models on 406-item data; haiku shown separately if desired
radar_models  = MODELS[:]
radar_scores  = {m: [SCORES[m]['REG'], SCORES[m]['NUM'], SCORES[m]['CON'], SCORES[m]['TMP']]
                 for m in radar_models}
radar_labels  = [
    'REG\n(Regulatory\nInterpretation)',
    'NUM\n(Numerical\nReasoning)',
    'CON\n(Contradiction\nDetection)',
    'TMP\n(Temporal\nReasoning)',
]

N      = 4
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [np.linspace(0, 2*np.pi, N, endpoint=False)[0]]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_facecolor('#f8f8f8')
fig.patch.set_facecolor('white')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for i, (model, col, mk) in enumerate(zip(radar_models, COLORS, MARKERS)):
    vals = radar_scores[model] + [radar_scores[model][0]]
    lw   = 2.5 if i < 3 else (2.0 if i < 7 else 1.5)
    ls   = '-'  if i < 7 else '--'
    ax.plot(angles, vals, marker=mk, color=col, linewidth=lw,
            linestyle=ls, markersize=5.5, label=f"{model} ({OVERALL[model]:.1f}%)",
            alpha=0.90, zorder=2)
    ax.fill(angles, vals, alpha=0.04, color=col, zorder=1)

ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11, fontweight='bold')
ax.set_ylim(50, 102)
ax.set_rgrids(
    [60, 70, 80, 90, 100],
    labels=['60%', '70%', '80%', '90%', '100%'],
    fontsize=8.5, color='grey', angle=0,
)
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.4)
ax.spines['polar'].set_visible(False)
ax.legend(
    loc='upper right', bbox_to_anchor=(1.55, 1.22),
    fontsize=8.5, framealpha=0.93, edgecolor='#cccccc',
    title='Model (Overall)', title_fontsize=9,
)
ax.set_title(
    'Model Capability Profiles Across Task Types\n(n = 406 items per model)',
    y=1.13, fontsize=13, fontweight='bold',
)
plt.tight_layout()
out = os.path.join(FIG_OUT, 'radar_chart.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out}")

# ── Figure 3: Difficulty Line Plot ────────────────────────────────────────

print("Generating Figure 3: Difficulty Line Plot…")

diff_data = [(m, DIFF_ACC[m]) for m in MODELS]
diff_data.sort(key=lambda x: -x[1].get('easy', 0))

# Difficulty counts for the full 406-item set
easy_n   = MODEL_DATA[MODELS[0]]['diff_total']['easy']
medium_n = MODEL_DATA[MODELS[0]]['diff_total']['medium']
hard_n   = MODEL_DATA[MODELS[0]]['diff_total']['hard']

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')
x = np.array([0, 1, 2])

for i, (model, col, mk) in enumerate(zip(MODELS, COLORS, MARKERS)):
    lw = 2.5 if i < 3 else (2.0 if i < 7 else 1.5)
    ls = '-'  if i < 7 else '--'
    d  = DIFF_ACC[model]
    ax.plot(
        x, [d['easy'], d['medium'], d['hard']],
        marker=mk, color=col, linewidth=lw, linestyle=ls,
        markersize=8, label=model, alpha=0.88,
    )

ax.axhline(60.0, color='#888', linestyle=':', linewidth=1.5,
           label='Human Expert (60.0%)', zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(
    [f'Easy\n(n={easy_n})', f'Medium\n(n={medium_n})', f'Hard\n(n={hard_n})'],
    fontsize=12,
)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(40, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.45, color='grey')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(
    loc='lower left', fontsize=8, ncol=2,
    framealpha=0.92, edgecolor='#cccccc',
    title='Model', title_fontsize=9,
)
ax.set_title(
    'Model Accuracy by Difficulty Level (Full 406-Item Evaluation)',
    fontsize=13, fontweight='bold', pad=12,
)
plt.tight_layout()
out = os.path.join(FIG_OUT, 'difficulty_lineplot.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out}")

# ── Figure 4: Inter-Task Spearman Correlation ─────────────────────────────

print("Generating Figure 4: Inter-Task Correlation…")

task_vecs = {
    'REG': [SCORES[m]['REG'] for m in MODELS],
    'NUM': [SCORES[m]['NUM'] for m in MODELS],
    'CON': [SCORES[m]['CON'] for m in MODELS],
    'TMP': [SCORES[m]['TMP'] for m in MODELS],
}
tasks_order = ['REG', 'NUM', 'CON', 'TMP']
corr_matrix = np.zeros((4, 4))
for i, t1 in enumerate(tasks_order):
    for j, t2 in enumerate(tasks_order):
        r, _ = spearmanr(task_vecs[t1], task_vecs[t2])
        corr_matrix[i, j] = r

task_full = ['Regulatory\nInterpretation', 'Numerical\nReasoning',
             'Contradiction\nDetection', 'Temporal\nReasoning']

fig, ax = plt.subplots(figsize=(6.5, 5.8))
fig.patch.set_facecolor('white')
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f',
    xticklabels=task_full, yticklabels=task_full,
    cmap='RdYlGn', vmin=-1, vmax=1, ax=ax,
    linewidths=0.5,
    cbar_kws={'label': 'Spearman ρ', 'shrink': 0.85},
    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
)
ax.set_xticklabels(task_full, fontsize=9, fontweight='bold', rotation=0, ha='center')
ax.set_yticklabels(task_full, fontsize=9, fontweight='bold', rotation=0)
ax.set_title(
    f'Inter-Task Spearman Correlation\n(model accuracy vectors, n={len(MODELS)} models)',
    fontsize=11, fontweight='bold', pad=10,
)
plt.tight_layout()
out = os.path.join(FIG_OUT, 'inter_task_correlation.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Summary report
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("VERIFIED LEADERBOARD (406-item evaluation)")
print("="*60)
print(f"{'Model':<22} {'REG':>6} {'NUM':>6} {'CON':>6} {'TMP':>6}  {'Overall':>8}  {'95% CI':>15}")
print("-"*75)
for model in MODELS:
    d  = wilson_results[model]
    pa = MODEL_DATA[model]['per_task_acc']
    print(
        f"{model:<22} {pa['REG']:>5.1f}% {pa['NUM']:>5.1f}% "
        f"{pa['CON']:>5.1f}% {pa['TMP']:>5.1f}%  "
        f"{MODEL_DATA[model]['overall_acc']:>6.1f}%  "
        f"[{d['overall']['ci_lo']:.1f}%, {d['overall']['ci_hi']:.1f}%]"
    )
print("-"*75)
hk = HAIKU_STATS
wh = wilson_results['Claude 3 Haiku†']
print(
    f"{'†Claude 3 Haiku':<22} {hk['REG']:>5.1f}% {hk['NUM']:>5.1f}% "
    f"{hk['CON']:>5.1f}% {hk['TMP']:>5.1f}%  "
    f"{hk['overall']:>6.1f}%  "
    f"[{wh['overall']['ci_lo']:.1f}%, {wh['overall']['ci_hi']:.1f}%]"
)
print("  † Evaluated on initial 150-item subset; not directly comparable to 406-item results.")

print("\nStatistically significant model pairs (p < 0.05, bootstrap 10k):")
for k, v in sorted(bootstrap_results.items(), key=lambda x: x[1]['p_value']):
    if v['significant']:
        print(f"  {v['model_a']} vs {v['model_b']}: Δ={v['delta']:+.1f}pp, p={v['p_value']:.4f}")

print("\nAll outputs saved. Done.\n")
