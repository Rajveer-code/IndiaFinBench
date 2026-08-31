"""Strict vs judge-corrected scoring: rank-shift figure + LaTeX table.

Everything is computed from evaluation/results_judged/*.csv. Nothing is hardcoded,
so the figure, the table, and the prose cannot drift apart.

Ranks use competition ranking (method='min'): tied scores share the better rank.
Three models tie at 95.6 and two at 96.1, so naive ordering would invent
distinctions the data does not support.

Outputs:
  paper/figures/figure_regime_shift.png / .pdf   (300 dpi)
  paper/tables/table_regime.tex
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

NAME = {'gemini': 'Gemini 2.5 Flash', 'gemini25_pro': 'Gemini 2.5 Pro', 'qwen3_32b': 'Qwen3-32B',
        'groq70b': 'LLaMA-3.3-70B', 'llama4scout': 'Llama 4 Scout 17B', 'kimi_k2': 'Kimi K2',
        'llama3': 'LLaMA-3-8B', 'gpt_oss_120b': 'GPT-OSS 120B', 'gpt_oss_20b': 'GPT-OSS 20B',
        'mistral': 'Mistral-7B', 'deepseek_r1_70b': 'DeepSeek R1 70B', 'gemma4_e4b': 'Gemma 4 E4B'}

rows = []
for path in glob.glob('evaluation/results_judged/*_results.csv'):
    key = os.path.basename(path).replace('_results.csv', '')
    if key not in NAME:
        continue
    d = pd.read_csv(path)
    final = d['auto_score'].astype(float).copy()
    final[(final == 0) & (d['judge_score'] == 1)] = 1
    rows.append({'model': NAME[key],
                 'strict': 100 * d['auto_score'].mean(),
                 'corrected': 100 * final.mean(),
                 'n': len(d)})

df = pd.DataFrame(rows).round({'strict': 1, 'corrected': 1})
assert (df['n'] == 406).all(), 'expected 406 items per model'
assert len(df) == 12, f'expected 12 models, got {len(df)}'

df['rank_s'] = df['strict'].rank(ascending=False, method='min').astype(int)
df['rank_c'] = df['corrected'].rank(ascending=False, method='min').astype(int)
df['shift'] = df['rank_s'] - df['rank_c']
df = df.sort_values('rank_s').reset_index(drop=True)

# ── Figure: rank-shift slopegraph ───────────────────────────────────────────
def label_offsets(rank_col):
    """Tied models share a rank, so their labels would print on top of each other.
    Spread each tied group symmetrically about the shared rank."""
    offsets = {}
    for _rank, group in df.groupby(rank_col):
        members = list(group.index)
        for position, idx in enumerate(members):
            offsets[idx] = ((position - (len(members) - 1) / 2) * 0.34
                            if len(members) > 1 else 0.0)
    return offsets


off_strict = label_offsets('rank_s')
off_corrected = label_offsets('rank_c')

fig, ax = plt.subplots(figsize=(7.6, 8.2))
for idx, r in df.iterrows():
    moved = abs(r['shift']) >= 2
    hero = r['model'] == 'DeepSeek R1 70B'
    colour = '#b2182b' if hero else ('#2166ac' if moved else '#b0b0b0')
    ax.plot([0, 1], [r['rank_s'], r['rank_c']], '-o', color=colour,
            lw=2.8 if hero else (1.8 if moved else 1.0),
            ms=8 if hero else 5, zorder=3 if hero else 2,
            alpha=1.0 if (hero or moved) else 0.7)
    ax.text(-0.04, r['rank_s'] + off_strict[idx],
            "{}  ({:.1f})".format(r['model'], r['strict']),
            ha='right', va='center', fontsize=9,
            fontweight='bold' if hero else 'normal', color=colour)
    ax.text(1.04, r['rank_c'] + off_corrected[idx],
            "({:.1f})  {}".format(r['corrected'], r['model']),
            ha='left', va='center', fontsize=9,
            fontweight='bold' if hero else 'normal', color=colour)

ax.set_xlim(-0.66, 1.66)
ax.set_ylim(12.7, 0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Strict\nstring matching', 'Corrected\njudge audit'],
                   fontsize=11, fontweight='bold')
ax.set_ylabel('Rank', fontsize=11)
ax.set_yticks(range(1, 13))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_title('The scoring regime reorders the leaderboard',
             fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
for ext in ('png', 'pdf'):
    plt.savefig('paper/figures/figure_regime_shift.' + ext, dpi=300, bbox_inches='tight')

# ── LaTeX table ─────────────────────────────────────────────────────────────
os.makedirs('paper/tables', exist_ok=True)
BS = chr(92)          # backslash, kept out of the literals below for clarity
ROW_END = BS + BS

lines = [BS + 'begin{tabular}{lrrrrr}',
         BS + 'toprule',
         'Model & Strict & Rank & Corrected & Rank & $' + BS + 'Delta$Rank ' + ROW_END,
         BS + 'midrule']
for _, r in df.iterrows():
    bold = r['model'] == 'DeepSeek R1 70B'

    def emph(text, _bold=bold):
        return (BS + 'textbf{' + text + '}') if _bold else text

    cells = [emph(r['model']),
             emph('{:.1f}'.format(r['strict'])),
             emph(str(r['rank_s'])),
             emph('{:.1f}'.format(r['corrected'])),
             emph(str(r['rank_c'])),
             emph('{:+d}'.format(r['shift']))]
    lines.append(' & '.join(cells) + ' ' + ROW_END)
lines += [BS + 'bottomrule', BS + 'end{tabular}']

with open('paper/tables/table_regime.tex', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')

print(df[['model', 'strict', 'rank_s', 'corrected', 'rank_c', 'shift']].to_string(index=False))
sub = df[df['model'] != 'Gemma 4 E4B']
print()
print('moved >= 2 ranks: {} of 12'.format(int((df['shift'].abs() >= 2).sum())))
print('strict spread    : {:.1f} pp'.format(df['strict'].max() - df['strict'].min()))
print('corrected spread : {:.1f} pp'.format(df['corrected'].max() - df['corrected'].min()))
print('excluding Gemma 4 E4B -> strict {:.1f} pp | corrected {:.1f} pp'.format(
    sub['strict'].max() - sub['strict'].min(),
    sub['corrected'].max() - sub['corrected'].min()))
print('corrected range: {:.1f} - {:.1f}'.format(df['corrected'].min(), df['corrected'].max()))
