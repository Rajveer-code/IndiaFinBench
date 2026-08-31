"""Phase 3 -- threshold sensitivity for the item-discrimination counts, and
near-duplicate detection over the full 406-item benchmark.

Threshold sensitivity: the paper's "137 highly-discriminative items" figure
depends on a chosen band (0.3 < p < 0.8, spread above median). Reporting only
that number invites the objection that it's a design choice dressed as a
fact. This shows how the count moves across neighbouring reasonable cutoffs,
so the reader can see the number is stable, not cherry-picked.

Near-duplicate detection: embeds every question+answer pair with the locally
installed nomic-embed-text model (free, no API) and reports pairs above a
cosine-similarity threshold. Better we report this ourselves than have a
reviewer find it.
"""
import json
import sys
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.novel_methods_utils import load_correctness_matrix  # noqa: E402

QA_PATH = Path("annotation/raw_qa/indiafinbench_qa_combined_406.json")

# ── Threshold sensitivity ───────────────────────────────────────────────────

matrix = load_correctness_matrix()
p = matrix.mean(axis=1)
spread = 4 * p * (1 - p)
median_spread = spread.median()

CEILING_CUTOFFS = [0.85, 0.90, 0.95]
FLOOR_CUTOFFS = [0.05, 0.10, 0.15]
BAND_CUTOFFS = [(0.25, 0.75), (0.30, 0.80), (0.35, 0.85)]

print("=== Ceiling-item count across cutoffs ===")
ceiling_table = {}
for c in CEILING_CUTOFFS:
    n = int((p >= c).sum())
    ceiling_table[c] = n
    print(f"  p >= {c}: {n} items ({100 * n / 406:.1f}%)")

print("\n=== Floor-item count across cutoffs ===")
floor_table = {}
for c in FLOOR_CUTOFFS:
    n = int((p <= c).sum())
    floor_table[c] = n
    print(f"  p <= {c}: {n} items")

print("\n=== Discriminative-band count across cutoffs (band AND spread > median) ===")
band_table = {}
for lo, hi in BAND_CUTOFFS:
    mask = (p > lo) & (p < hi) & (spread > median_spread)
    n = int(mask.sum())
    band_table[f"{lo}-{hi}"] = n
    print(f"  {lo} < p < {hi}, spread > median: {n} items")

sensitivity = {
    "ceiling_by_cutoff": {str(k): v for k, v in ceiling_table.items()},
    "floor_by_cutoff": {str(k): v for k, v in floor_table.items()},
    "discriminative_band_by_cutoff": band_table,
    "median_spread": round(float(median_spread), 4),
    "paper_default_ceiling_0.90": int((p >= 0.90).sum()),
    "paper_default_floor_0.10": int((p <= 0.10).sum()),
    "paper_default_discriminative_0.3_0.8": band_table["0.3-0.8"],
}
with open("evaluation/threshold_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(sensitivity, f, indent=2)
print("\nSaved evaluation/threshold_sensitivity.json")

# ── Near-duplicate detection via nomic-embed-text ───────────────────────────

with open(QA_PATH, encoding="utf-8") as f:
    items = json.load(f)

texts, ids = [], []
for it in items:
    q = it.get("question", "")
    a = it.get("answer", "")
    texts.append(f"{q} {a}".strip())
    ids.append(it["id"])


def embed(text: str) -> list[float]:
    payload = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embeddings", data=payload,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["embedding"]


print(f"\nEmbedding {len(texts)} items via nomic-embed-text...")
vecs = np.zeros((len(texts), 768))
for i, t in enumerate(texts):
    vecs[i] = embed(t)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(texts)}")

norms = np.linalg.norm(vecs, axis=1, keepdims=True)
unit = vecs / norms
sim = unit @ unit.T
np.fill_diagonal(sim, -1)

THRESH = 0.95
pairs = []
for i, j in combinations(range(len(ids)), 2):
    if sim[i, j] >= THRESH:
        pairs.append((ids[i], ids[j], round(float(sim[i, j]), 4)))
pairs.sort(key=lambda x: -x[2])

print(f"\n=== Near-duplicate pairs (cosine >= {THRESH}) ===")
print(f"  {len(pairs)} pairs found out of {len(ids) * (len(ids) - 1) // 2} total")
for a, b, s in pairs[:20]:
    print(f"  {a:10s} {b:10s} {s:.4f}")

with open("evaluation/near_duplicate_pairs.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": THRESH, "n_pairs": len(pairs),
               "n_total_pairs": len(ids) * (len(ids) - 1) // 2,
               "pairs": pairs}, f, indent=2)
print(f"\nSaved evaluation/near_duplicate_pairs.json")
