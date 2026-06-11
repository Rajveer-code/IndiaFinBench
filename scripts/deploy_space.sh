#!/usr/bin/env bash
# Deploy the IndiaFinBench demo to HuggingFace Spaces.
#
# The Space repo is a filtered artifact of this repository: only the files
# the Docker build needs (Dockerfile, demo/, rag/ with the production index),
# with binaries stored via Git LFS as HF requires. Research artifacts
# (paper/, evaluation/, results/, annotation/) stay on GitHub only.
#
# Usage: bash scripts/deploy_space.sh
set -euo pipefail

SPACE_URL="https://huggingface.co/spaces/Rajveer-code/IndiaFinBench"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Assembling deploy tree in $TMP"
cp "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$TMP/"
cp -r "$ROOT/demo" "$TMP/demo"
cp -r "$ROOT/rag" "$TMP/rag"

# Prune everything the build does not need
rm -rf "$TMP"/demo/__pycache__ "$TMP"/demo/*/__pycache__ "$TMP"/demo/.claude
rm -rf "$TMP"/rag/__pycache__ "$TMP"/rag/*/__pycache__
rm -rf "$TMP"/rag/index_800 "$TMP"/rag/index_2400   # ablation indexes
rm -f  "$TMP"/demo/leaderboard.db                    # seeded at startup

# Space README: HF frontmatter + pointer back to the canonical repo
cat > "$TMP/README.md" <<'EOF'
---
title: IndiaFinBench
emoji: 📜
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: LLM benchmark for Indian financial regulation
---

# IndiaFinBench

**The first evaluation benchmark for large language model performance on Indian financial regulatory text.**

406 expert-annotated questions over 192 SEBI & RBI regulatory documents (1992–2026) · 12 frontier models evaluated · hybrid FAISS + BM25 retrieval with Recall@5 = 0.785.

This Space hosts the live research site: the full leaderboard with statistical tier analysis, a dataset explorer, a live hybrid-RAG demo over the regulatory corpus, and model submission.

- **Dataset:** [Rajveer-code/IndiaFinBench](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench) (CC BY 4.0)
- **Code & paper:** [github.com/Rajveer-code/IndiaFinBench](https://github.com/Rajveer-code/IndiaFinBench) (MIT)
- **Author:** Rajveer Singh Pall — rajveerpall04@gmail.com
EOF

cd "$TMP"
git init -b main -q
git config user.name  "Rajveer Singh Pall"
git config user.email "rajveerpall04@gmail.com"
git lfs install --local >/dev/null
git lfs track "*.pkl" "*.index" >/dev/null
git add -A
git commit -q -m "Deploy IndiaFinBench research site"
git push --force "$SPACE_URL" main

echo "Deployed. Build status: $SPACE_URL"
