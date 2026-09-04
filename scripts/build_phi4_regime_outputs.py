"""Build the three-regime rank-shift figure from evaluation/regime_three_way.json.

Plan v3 Phase 4.2: rebuilt as a three-column plot (strict -> judge-only -> judge-augmented).
Judge-only is this paper's second primary regime; judge-augmented is a sensitivity regime,
shown in a lighter tone and thinner lines throughout.

Table generation was REMOVED from this script (2026-09-02): it used to write
paper/tables/table_regime.tex directly from a 2-regime computation, which
silently conflicted with scripts/regime_table.py's 3-regime output to the
same file -- whichever ran last would win, with no error. regime_table.py is
now the single source of truth for that table; run it separately. This script
only reads its (already-verified) numbers and only produces the figure.

Uses competition ranking (ties share the better rank) -- a prior version of
this exact computation used plain positional ranking and silently broke a
genuine exact tie (DeepSeek-R1-Distill / LLaMA-3.3-70B, both 398/406).
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

d = json.load(open("evaluation/regime_three_way.json", encoding="utf-8"))
per = d["per_model"]
models = list(per.keys())

df = pd.DataFrame({"model": models}).set_index("model")
df["strict"] = pd.Series({m: per[m]["strict_pct"] for m in models})
df["judge_only"] = pd.Series({m: per[m]["judge_only_pct"] for m in models})
df["augmented"] = pd.Series({m: per[m]["judge_augmented_pct"] for m in models})
df["rank_s"] = pd.Series({m: per[m]["strict_rank"] for m in models})
df["rank_j"] = pd.Series({m: per[m]["judge_only_rank"] for m in models})
df["rank_a"] = pd.Series({m: per[m]["judge_augmented_rank"] for m in models})
df["shift_sj"] = df["rank_s"] - df["rank_j"]
df = df.sort_values("rank_s")


def label_offsets(rank_col, spacing=0.34):
    offsets = {}
    for _, group in df.groupby(rank_col):
        members = list(group.index)
        for i, idx in enumerate(members):
            offsets[idx] = (i - (len(members) - 1) / 2) * spacing if len(members) > 1 else 0.0
    return offsets


off_s = label_offsets("rank_s")
# Judge-only labels are small font (7pt) and short ("(NN.N)"), so a tied pair at the
# default 0.34 spacing (±0.17) renders as two nearly-overlapping, illegible strings --
# confirmed on the Qwen3-32B / DeepSeek-R1-Distill exact tie at 92.36%. Wider spacing here only.
off_j = label_offsets("rank_j", spacing=0.85)
off_a = label_offsets("rank_a")

fig, ax = plt.subplots(figsize=(9.5, 8.2))
for m, r in df.iterrows():
    moved = abs(r["shift_sj"]) >= 2
    hero = m == "DeepSeek-R1-Distill"
    colour = "#b2182b" if hero else ("#2166ac" if moved else "#b0b0b0")
    lw_primary = 2.8 if hero else (1.8 if moved else 1.0)
    ms_primary = 8 if hero else 5
    alpha_primary = 1.0 if (hero or moved) else 0.7
    # Strict -> judge-only: the primary-regime comparison, full weight.
    ax.plot([0, 1], [r.rank_s, r.rank_j], "-o", color=colour, lw=lw_primary,
            ms=ms_primary, zorder=3 if hero else 2, alpha=alpha_primary)
    # Judge-only -> judge-augmented: the sensitivity regime, lighter/thinner throughout.
    ax.plot([1, 2], [r.rank_j, r.rank_a], "o", color=colour, lw=lw_primary * 0.55,
            ms=ms_primary * 0.7, zorder=1, alpha=alpha_primary * 0.45, linestyle=(0, (4, 2)))
    ax.text(-0.05, r.rank_s + off_s[m], f"{m}  ({r.strict:.1f})",
            ha="right", va="center", fontsize=8.5,
            fontweight="bold" if hero else "normal", color=colour)
    ax.text(1.0, r.rank_j + off_j[m] - 0.02, f"({r.judge_only:.1f})",
            ha="center", va="bottom", fontsize=7, color=colour, alpha=0.9)
    ax.text(2.05, r.rank_a + off_a[m], f"({r.augmented:.1f})  {m}",
            ha="left", va="center", fontsize=8.5,
            fontweight="bold" if hero else "normal", color=colour, alpha=0.75)

ax.set_xlim(-1.05, 2.85)
ax.set_ylim(12.7, 0.3)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Strict\nstring matching", "Judge-only\n(primary)",
                     "Judge-augmented\n(sensitivity)"], fontsize=10.5, fontweight="bold")
ax.get_xticklabels()[2].set_alpha(0.6)
ax.set_ylabel("Rank")
ax.set_yticks(range(1, 13))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_title("The scoring regime reorders the leaderboard", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"paper/figures/figure_regime_shift.{ext}", dpi=300, bbox_inches="tight")

print(df[["strict", "rank_s", "judge_only", "rank_j", "augmented", "rank_a", "shift_sj"]].to_string())
print(f"\nstrict<->judge-only moved >=2 ranks: {(df['shift_sj'].abs() >= 2).sum()} of 12")
print(f"strict range: {df.strict.min():.2f}-{df.strict.max():.2f}")
print(f"judge-only range: {df.judge_only.min():.2f}-{df.judge_only.max():.2f}")
print(f"judge-augmented range: {df.augmented.min():.2f}-{df.augmented.max():.2f}")
