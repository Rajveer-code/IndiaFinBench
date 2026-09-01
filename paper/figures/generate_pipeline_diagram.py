"""Regenerates the dataset-construction pipeline diagram (Figure 2).

No generator script existed for the original fig_pipeline.png -- it was a
static image that went stale (still said "120-item evaluation, kappa=0.611,
76.7% overall" for the Human IAA box, while the actual IAA study is 180
items, kappa=0.645, 77.2%, per Table 1 / draft_07 Section 3.5). Recreated
from scratch matching the original's box/arrow layout and colour scheme.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

boxes = [
    ("Source documents", "[192 SEBI + RBI PDFs\n1992–2026]", "#e0937a"),
    ("PDF parsing", "pdfplumber\ntext + tables", "#e8b04b"),
    ("Expert\nannotation", "REG 174 items\nNUM 92 items\nCON 62 items\nTMP 78 items\n\n406 total", "#b48ec4"),
    ("Quality\nvalidation", "Model-based\nsecondary pass\nκ = 0.918 (CON)\n90.7% overall", "#5fada0"),
    ("Human IAA", "180-item\nevaluation\n\nκ = 0.645 (CON)\n77.2% overall", "#5b8fc9"),
    ("IndiaFinBench", "406 QA pairs\n4 task types\n12 models\nevaluated", "#5a9e6f"),
]

fig, ax = plt.subplots(figsize=(11, 4.2))
n = len(boxes)
box_w, box_h = 1.5, 2.6
gap = 0.55
x0 = 0.3

centers = []
for i, (title, body, color) in enumerate(boxes):
    x = x0 + i * (box_w + gap)
    y = 0.5
    centers.append((x + box_w / 2, y + box_h / 2))
    box = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.12",
                          facecolor=color, edgecolor="none", zorder=2)
    ax.add_patch(box)
    ax.text(x + box_w / 2, y + box_h - 0.35, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="black", zorder=3)
    ax.text(x + box_w / 2, y + box_h - 0.75, body, ha="center", va="top",
            fontsize=9.5, color="black", zorder=3, linespacing=1.5)

for i in range(n - 1):
    x_start = x0 + i * (box_w + gap) + box_w
    x_end = x_start + gap
    y = 0.5 + box_h / 2
    arrow = FancyArrowPatch((x_start + 0.05, y), (x_end - 0.05, y),
                             arrowstyle="-|>", mutation_scale=18,
                             color="#c9922e", linewidth=2, zorder=1)
    ax.add_patch(arrow)

ax.set_xlim(0, x0 + n * (box_w + gap))
ax.set_ylim(0, 3.6)
ax.axis("off")
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"paper/figures/fig_pipeline.{ext}", dpi=300, bbox_inches="tight")
print("Saved paper/figures/fig_pipeline.png/pdf")
