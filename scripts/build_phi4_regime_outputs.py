"""Build the final regime table + figure from the completed phi4-mini judge.

Replaces the earlier Gemini-judge-based generate_regime_figure.py output.
Uses competition ranking (ties share the better rank) -- a prior version of
this exact computation used plain positional ranking and silently broke a
genuine exact tie (DeepSeek-R1-Distill / LLaMA-3.3-70B, both 398/406).
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

d = json.load(open("evaluation/phi4_regime_table.json", encoding="utf-8"))
per = d["per_model"]
models = list(per.keys())


def competition_rank(vals: dict) -> dict:
    order = sorted(set(vals.values()), reverse=True)
    rank_of = {v: i + 1 for i, v in enumerate(order)}
    return {m: rank_of[v] for m, v in vals.items()}


strict = {m: per[m]["strict_accuracy_406"] for m in models}
phi4 = {m: per[m]["phi4_corrected_accuracy_406"] for m in models}
sr = competition_rank(strict)
pr = competition_rank(phi4)

df = pd.DataFrame({"model": models}).set_index("model")
df["strict"] = pd.Series(strict)
df["phi4"] = pd.Series(phi4)
df["rank_s"] = pd.Series(sr)
df["rank_c"] = pd.Series(pr)
df["shift"] = df["rank_s"] - df["rank_c"]
df = df.sort_values("rank_s")

# ── LaTeX table ─────────────────────────────────────────────────────────────
BS = chr(92)
lines = [BS + "begin{tabular}{lrrrrr}", BS + "toprule",
         "Model & Strict & Rank & Judge-audited & Rank & $" + BS + "Delta$Rank " + BS + BS,
         BS + "midrule"]
for m, r in df.iterrows():
    bold = m == "DeepSeek-R1-Distill"
    def e(s, b=bold):
        return (BS + "textbf{" + s + "}") if b else s
    lines.append(" & ".join([e(m), e(f"{r.strict:.1f}"), e(str(int(r.rank_s))),
                              e(f"{r.phi4:.1f}"), e(str(int(r.rank_c))),
                              e(f"{int(r['shift']):+d}")]) + " " + BS + BS)
lines += [BS + "bottomrule", BS + "end{tabular}"]
open("paper/tables/table_regime.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ── Figure ──────────────────────────────────────────────────────────────────
def label_offsets(rank_col):
    offsets = {}
    for _, group in df.groupby(rank_col):
        members = list(group.index)
        for i, idx in enumerate(members):
            offsets[idx] = (i - (len(members) - 1) / 2) * 0.34 if len(members) > 1 else 0.0
    return offsets


off_s = label_offsets("rank_s")
off_c = label_offsets("rank_c")

fig, ax = plt.subplots(figsize=(7.8, 8.2))
for m, r in df.iterrows():
    moved = abs(r["shift"]) >= 2
    hero = m == "DeepSeek-R1-Distill"
    colour = "#b2182b" if hero else ("#2166ac" if moved else "#b0b0b0")
    ax.plot([0, 1], [r.rank_s, r.rank_c], "-o", color=colour,
            lw=2.8 if hero else (1.8 if moved else 1.0),
            ms=8 if hero else 5, zorder=3 if hero else 2,
            alpha=1.0 if (hero or moved) else 0.7)
    ax.text(-0.04, r.rank_s + off_s[m], f"{m}  ({r.strict:.1f})",
            ha="right", va="center", fontsize=9,
            fontweight="bold" if hero else "normal", color=colour)
    ax.text(1.04, r.rank_c + off_c[m], f"({r.phi4:.1f})  {m}",
            ha="left", va="center", fontsize=9,
            fontweight="bold" if hero else "normal", color=colour)

ax.set_xlim(-0.7, 1.7)
ax.set_ylim(12.7, 0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Strict\nstring matching", "Judge-audited\n(cross-model)"],
                    fontsize=11, fontweight="bold")
ax.set_ylabel("Rank")
ax.set_yticks(range(1, 13))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_title("The scoring regime reorders the leaderboard", fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"paper/figures/figure_regime_shift.{ext}", dpi=300, bbox_inches="tight")

print(df[["strict", "rank_s", "phi4", "rank_c", "shift"]].to_string())
print(f"\nmoved >=2: {(df['shift'].abs() >= 2).sum()} of 12")
print(f"strict range: {df.strict.min():.1f}-{df.strict.max():.1f}")
print(f"phi4 range: {df.phi4.min():.2f}-{df.phi4.max():.2f}")
