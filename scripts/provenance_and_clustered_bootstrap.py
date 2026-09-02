"""Phase 1 -- provenance audit and document-clustered significance testing.

Why this exists: the paper claims 406 items are "drawn from 192 SEBI and RBI
documents." 192 is the size of the *collected corpus*; the QA items actually
trace to 34 documents (verified by exhaustive scan of the annotation JSON).
That means items are not independent draws -- several questions can share a
document, and item-level bootstrap CIs / p-values are optimistically narrow.

Cluster construction: a graph over the 34 documents, with an edge for every
CON item linking document_a-document_b (CON items depend on two documents at
once). Connected components of this graph are the resampling units. This is
stricter than clustering CON on the unordered pair alone: a REG item on
document A and a CON item on (A, B) are dependent, and putting them in the
same component is what keeps clusters disjoint while respecting that
dependency -- naive pair-clustering would let a REG item and a CON pair that
shares document A land in two different "independent" clusters, which is
exactly the kind of leakage a cluster bootstrap exists to prevent.

Outputs:
  evaluation/provenance_audit.csv         -- item_id -> document(s) -> cluster
  evaluation/provenance_summary.json      -- collected vs represented doc counts
  evaluation/clustered_bootstrap.json     -- item-level vs component-clustered
                                              p-values for the headline pairs
  evaluation/clustered_bootstrap_full66.json -- same, for all 66 model pairs
                                              (the earlier copy of this file, dated
                                              2026-08-31, had no generating script at
                                              all -- see Plan v3 Phase 3.3)

Plan v3 Phase 3.3: the paper's own record noted the Bonferroni-survivor count moved
between 15 and 16 across independent 10,000-resample runs -- a Monte-Carlo-unstable
integer is indefensible in a paper about measurement precision. N_RESAMPLES defaults
to 100,000. Both N_RESAMPLES and SEED are overridable via environment variables
(BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED) so a stability sweep across several seeds
does not require editing this file -- see scripts/bootstrap_seed_sweep.py.
"""
import csv
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.novel_methods_utils import MODEL_FILES, RESULTS_DIR  # noqa: E402

QA_PATH = Path("annotation/raw_qa/indiafinbench_qa_combined_406.json")
N_RESAMPLES = int(os.environ.get("BOOTSTRAP_N_RESAMPLES", 100_000))
SEED = int(os.environ.get("BOOTSTRAP_SEED", 42))
OUT_SUFFIX = os.environ.get("BOOTSTRAP_OUT_SUFFIX", "")  # e.g. "_seed7" for sweep runs

# ── 1. Load items, build the document graph ─────────────────────────────────

with open(QA_PATH, encoding="utf-8") as f:
    items = json.load(f)
assert len(items) == 406

# Union-Find over documents.
parent: dict[str, str] = {}


def find(x: str) -> str:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a: str, b: str) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


item_docs: dict[str, list[str]] = {}
for it in items:
    docs = []
    if it.get("document"):
        docs.append(it["document"])
    if it.get("document_a"):
        docs.append(it["document_a"])
    if it.get("document_b"):
        docs.append(it["document_b"])
    item_docs[it["id"]] = docs
    for d in docs:
        find(d)  # register node
    if len(docs) == 2:
        union(docs[0], docs[1])

all_docs = sorted(parent.keys())
components: dict[str, list[str]] = defaultdict(list)
for d in all_docs:
    components[find(d)].append(d)

doc_to_cluster = {d: find(d) for d in all_docs}
cluster_ids = {root: i for i, root in enumerate(sorted(components.keys()))}

item_cluster: dict[str, int] = {}
for iid, docs in item_docs.items():
    assert docs, f"{iid} has no document reference"
    roots = {doc_to_cluster[d] for d in docs}
    assert len(roots) == 1, f"{iid} spans multiple components -- clustering bug"
    item_cluster[iid] = cluster_ids[roots.pop()]

assert len(item_cluster) == 406, f"only {len(item_cluster)}/406 items got a cluster"
assert set(item_cluster.keys()) == {it["id"] for it in items}

# ── 2. Provenance audit CSV + summary ───────────────────────────────────────

by_task = {it["id"]: it["task_type"] for it in items}
by_diff = {it["id"]: it.get("difficulty", "") for it in items}

with open("evaluation/provenance_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "task_type", "difficulty", "document(s)", "cluster_id"])
    for iid in sorted(item_docs):
        w.writerow([iid, by_task[iid], by_diff[iid],
                    ";".join(item_docs[iid]), item_cluster[iid]])

doc_item_counts = defaultdict(int)
for iid, docs in item_docs.items():
    for d in set(docs):
        doc_item_counts[d] += 1

cluster_sizes = defaultdict(int)
for iid, c in item_cluster.items():
    cluster_sizes[c] += 1
sizes = sorted(cluster_sizes.values(), reverse=True)

summary = {
    "documents_collected_corpus": 192,  # 92 SEBI + 100 RBI, per metadata_*.csv
    "documents_represented_in_qa": len(all_docs),
    "connected_components": len(components),
    "largest_document_item_count": max(doc_item_counts.values()),
    "median_document_item_count": sorted(doc_item_counts.values())[len(doc_item_counts) // 2],
    "largest_component_size": sizes[0],
    "median_component_size": sizes[len(sizes) // 2],
    "n_components_size_1": sum(1 for s in sizes if s == 1),
    "n_components_size_gte_10": sum(1 for s in sizes if s >= 10),
    "component_sizes_sorted_desc": sizes,
}
with open("evaluation/provenance_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=== Provenance summary ===")
for k, v in summary.items():
    if k != "component_sizes_sorted_desc":
        print(f"  {k}: {v}")
print(f"  component sizes: {sizes}")

# ── 3. Correctness checks (valid ones -- NOT "clustered CI must be wider") ──

assert sum(len(v) for v in components.values()) == len(all_docs), "components not disjoint"
seen = set()
for c, docs in components.items():
    for d in docs:
        assert d not in seen, f"document {d} appears in two components"
        seen.add(d)
assert seen == set(all_docs), "not every document assigned to a component"
print("\nCluster-assignment checks: PASS (disjoint, exhaustive, no leakage)")

# ── 4. Load per-item correctness for all 12 models ──────────────────────────

model_correct: dict[str, dict[str, int]] = {}
for label, fname in MODEL_FILES.items():
    path = RESULTS_DIR / fname
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    model_correct[label] = {r["id"]: int(r["correct"]) for r in rows}

item_ids_sorted = sorted(item_cluster.keys())
cluster_of = np.array([item_cluster[i] for i in item_ids_sorted])
unique_clusters = np.unique(cluster_of)


def vec(model: str) -> np.ndarray:
    return np.array([model_correct[model][i] for i in item_ids_sorted])


# ── 5. Bootstrap: item-level vs cluster (connected-component) resampling ───

rng = np.random.default_rng(SEED)


def bootstrap_item(a: np.ndarray, b: np.ndarray, n=N_RESAMPLES) -> float:
    diff = a - b
    obs = diff.mean()
    centred = diff - diff.mean()
    n_items = len(centred)
    count = 0
    for _ in range(n):
        sample = centred[rng.integers(0, n_items, n_items)]
        if abs(sample.mean()) >= abs(obs):
            count += 1
    return count / n


def bootstrap_cluster(a: np.ndarray, b: np.ndarray, n=N_RESAMPLES) -> float:
    """Resample clusters with replacement; each resample keeps every item in
    the drawn clusters (standard cluster bootstrap), not one item per draw."""
    diff = a - b
    obs = diff.mean()
    cluster_diff_lists = [diff[cluster_of == c] for c in unique_clusters]
    n_clusters = len(unique_clusters)
    # Centre at the cluster level so the null mean is 0.
    grand_mean = diff.mean()
    count = 0
    for _ in range(n):
        draw = rng.integers(0, n_clusters, n_clusters)
        sample = np.concatenate([cluster_diff_lists[k] for k in draw]) - grand_mean
        if abs(sample.mean()) >= abs(obs - grand_mean + grand_mean):
            count += 1
    return count / n


HEADLINE_PAIRS = [
    ("Gemini 2.5 Flash", "Qwen3-32B"),
    ("Gemini 2.5 Flash", "Gemma 3 4B"),
    ("Llama 4 Scout 17B", "LLaMA-3.3-70B"),
    ("GPT-OSS 120B", "GPT-OSS 20B"),
    ("Gemma 3 4B", "DeepSeek R1 70B"),
    ("Gemma 3 4B", "Mistral-7B"),
]
ALL_PAIRS = list(itertools.combinations(sorted(MODEL_FILES.keys()), 2))
assert len(ALL_PAIRS) == 66, f"expected C(12,2)=66 pairs, got {len(ALL_PAIRS)}"
alpha_bonf = 0.05 / len(ALL_PAIRS)

full_results = {}
print(f"\n=== All 66 pairs: item-level vs component-clustered p-values "
      f"(N={N_RESAMPLES:,}, seed={SEED}) ===")
for ma, mb in ALL_PAIRS:
    va, vb = vec(ma), vec(mb)
    p_item = bootstrap_item(va, vb)
    p_clus = bootstrap_cluster(va, vb)
    key = f"{ma} vs {mb}"
    full_results[key] = {
        "acc_a": round(100 * va.mean(), 1), "acc_b": round(100 * vb.mean(), 1),
        "p_item_level": round(p_item, 4), "p_cluster": round(p_clus, 4),
        "sig_item_level_05": p_item < 0.05, "sig_cluster_05": p_clus < 0.05,
        "sig_item_level_bonf": p_item < alpha_bonf, "sig_cluster_bonf": p_clus < alpha_bonf,
        "flips_direction": (p_item < 0.05) != (p_clus < 0.05),
    }

n_sig_item = sum(1 for r in full_results.values() if r["sig_item_level_05"])
n_sig_cluster = sum(1 for r in full_results.values() if r["sig_cluster_05"])
n_bonf_item = sum(1 for r in full_results.values() if r["sig_item_level_bonf"])
n_bonf_cluster = sum(1 for r in full_results.values() if r["sig_cluster_bonf"])
n_flips = sum(1 for r in full_results.values() if r["flips_direction"])
print(f"  item-level: {n_sig_item}/66 sig at p<0.05, {n_bonf_item}/66 survive Bonferroni "
      f"(alpha={alpha_bonf:.5f})")
print(f"  clustered:  {n_sig_cluster}/66 sig at p<0.05, {n_bonf_cluster}/66 survive Bonferroni")
print(f"  {n_flips}/66 pairs flip significance direction (p<0.05) between item and cluster level")

def _lookup(ma: str, mb: str) -> tuple[str, dict]:
    """ALL_PAIRS keys are alphabetically ordered; HEADLINE_PAIRS is not."""
    key = f"{ma} vs {mb}"
    if key in full_results:
        return key, full_results[key]
    return f"{mb} vs {ma}", full_results[f"{mb} vs {ma}"]


results = {}
for ma, mb in HEADLINE_PAIRS:
    key, r = _lookup(ma, mb)
    results[f"{ma} vs {mb}"] = {
        "acc_a": r["acc_a"] if key == f"{ma} vs {mb}" else r["acc_b"],
        "acc_b": r["acc_b"] if key == f"{ma} vs {mb}" else r["acc_a"],
        "p_item_level": r["p_item_level"], "p_cluster": r["p_cluster"],
        "sig_item_level": r["sig_item_level_05"], "sig_cluster": r["sig_cluster_05"],
    }
print("\n=== Headline pairs (subset of the above, for the main-text table) ===")
for key, r in results.items():
    print(f"  {key:45s} item={r['p_item_level']:.4f}  cluster={r['p_cluster']:.4f}  "
          f"{'AGREE' if r['sig_item_level'] == r['sig_cluster'] else '<<< CONCLUSION CHANGES'}")

with open("evaluation/clustered_bootstrap.json", "w", encoding="utf-8") as f:
    json.dump({"n_resamples": N_RESAMPLES, "seed": SEED,
               "n_components": len(components), "component_sizes": sizes,
               "pairs": results}, f, indent=2)

with open(f"evaluation/clustered_bootstrap_full66{OUT_SUFFIX}.json", "w", encoding="utf-8") as f:
    json.dump({"n_resamples": N_RESAMPLES, "seed": SEED, "n_pairs": 66,
               "alpha_bonferroni": round(alpha_bonf, 6),
               "n_sig_item_05": n_sig_item, "n_sig_cluster_05": n_sig_cluster,
               "n_sig_item_bonferroni": n_bonf_item, "n_sig_cluster_bonferroni": n_bonf_cluster,
               "n_pairs_flip_direction_05": n_flips,
               "n_components": len(components), "component_sizes": sizes,
               "pairs": full_results}, f, indent=2)

print(f"\nSaved evaluation/provenance_audit.csv, evaluation/provenance_summary.json, "
      f"evaluation/clustered_bootstrap.json, evaluation/clustered_bootstrap_full66{OUT_SUFFIX}.json")

# ── 6. Determinism check (fixed seed -> identical output on rerun) ─────────
rng2 = np.random.default_rng(SEED)
ma, mb = HEADLINE_PAIRS[0]
va, vb = vec(ma), vec(mb)


def bootstrap_item_check(a, b, rng_local, n=1000):
    diff = a - b
    obs = diff.mean()
    centred = diff - diff.mean()
    n_items = len(centred)
    count = 0
    for _ in range(n):
        sample = centred[rng_local.integers(0, n_items, n_items)]
        if abs(sample.mean()) >= abs(obs):
            count += 1
    return count / n


check_a = bootstrap_item_check(va, vb, np.random.default_rng(SEED))
check_b = bootstrap_item_check(va, vb, np.random.default_rng(SEED))
print(f"Determinism check (fixed seed, 1000 resamples, run twice): "
      f"{'PASS' if check_a == check_b else 'FAIL'} ({check_a} == {check_b})")
