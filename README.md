<div align="center">

# IndiaFinBench

### The First Evaluation Benchmark for Large Language Models on Indian Financial Regulatory Text

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-HuggingFace_Spaces-FFD21E?style=flat-square)](https://huggingface.co/spaces/Rajveer-code/IndiaFinBench)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-Rajveer--code%2FIndiaFinBench-FFD21E?style=flat-square)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19298-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.19298)
[![Target venue: TMLR](https://img.shields.io/badge/Target_venue-TMLR-blue?style=flat-square)](https://jmlr.org/tmlr/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21739533.svg)](https://doi.org/10.5281/zenodo.21739533)

<br>

| 406 | 192 | 12 | 0.785 | 80.0% |
|:---:|:---:|:---:|:---:|:---:|
| Expert-annotated QA items | SEBI + RBI documents | LLMs benchmarked | Hybrid RAG Recall@5 | Human reference (n=60, med+hard) |

</div>

---

## Overview

A benchmark leaderboard reports one accuracy number, but that number is a joint product of a model **and** a scoring rule. Score the same 406 predictions from the same twelve models under strict string matching versus an LLM judge's own verdict, and the two rankings show **no positive rank correspondence** (Spearman ρ = −0.273, p = 0.39) — not "mostly agree with some noise," but no significant relationship at all. One model ranks last under strict scoring and ties for first once a judge is allowed to rescue (never reject) its answers.

IndiaFinBench is the instrument that makes this measurable, not the headline claim itself. It is a **zero-shot, closed-book evaluation benchmark** of 406 expert-annotated question-answer pairs drawn from 192 circulars, regulations, and master directions published by SEBI and the RBI between 1992 and 2026 — a domain chosen specifically because it's underrepresented in the training data most benchmarks implicitly assume. Four task types (regulatory interpretation, numerical reasoning, contradiction detection, temporal reasoning) must be solved from document evidence alone, without retrieval.

Beyond the scoring-regime finding: efficiency dominates scale (a 17B-parameter model statistically matches a 70B model on this benchmark), item-level analysis shows over a third of the benchmark's items separate no pair of the twelve evaluated systems at all, and on a 60-item medium-and-hard subset no model significantly outperforms a careful non-specialist human (80.0%) once corrected for testing twelve models at once.

---

## Research Contributions

1. **Scoring-regime sensitivity, measured not asserted** — the same 406 predictions from 12 LLMs scored under strict string matching and under a full-coverage cross-model LLM judge (every prediction judged, not only strict failures) show no positive rank correspondence (Spearman ρ = −0.273, p = 0.39). A third, asymmetric composite regime (rescue-but-never-reject) illustrates how far one specific, non-authoritative scoring choice can move a model.

2. **A direct test of the verbosity explanation, not an assumption** — it's tempting to attribute judge overturns to output length; tested directly rather than asserted, and response length does not predict which strict failures a judge reclassifies once task type and model identity are controlled for. Reported as the negative result it is.

3. **Discriminative-coverage analysis** — an exact pairwise-disagreement measure (D(k) = k(12−k) over 12 models) shows 148 of 406 items (36.5%) separate no pair of the evaluated systems at all. Nominal benchmark size and discriminative coverage are different quantities, rarely distinguished in benchmark reporting.

4. **IndiaFinBench itself** — 406 expert-annotated items across four task types, sourced from 34 of 192 collected SEBI and RBI documents (1992–2026), with a validated annotation protocol (90.7% model-based agreement, 86.1% human IAA over 180 items).

5. **A human reference point reported with its limits stated** — a single non-specialist evaluator on a 60-item medium/hard subset (80.0%, 95% Wilson CI [68.2%, 88.2%]); paired bootstrap tests against all twelve models find no significant difference in either direction once corrected for the twelve-test family.

6. **Full public release** — dataset, evaluation harness, per-item judge verdicts, and all model predictions, so every regime and every claim can be reproduced or contested independently.

7. **Hybrid RAG system with full ablation** (repository extra, not part of the paper's claims) — production-grade FAISS + BM25 pipeline with Reciprocal Rank Fusion, benchmarked across six retrieval configurations. Hybrid RRF improves Recall@5 by +9.7pp over dense-only retrieval; optimal chunk size empirically determined at 1,600 characters.

---

## Key Findings

> **Strict scoring and a cross-model judge's own verdict rank the field with no positive correspondence.** The same 406-item predictions, scored under strict string matching versus a cross-model judge (phi4-mini, sharing no model family with any of the 12 evaluated systems)'s own verdict taken as final in both directions — not merely accepting judge overturns — give Spearman ρ = −0.273 (p = 0.39) across the twelve models: only one model holds its rank under both. A third, asymmetric regime (strict-OR-judge — rescue failures, never challenge successes) is a sensitivity check, not a primary result: under it, DeepSeek-R1-Distill-Llama-70B goes from last under strict scoring to a tie for 1st, because the judge reclassifies 93.9% of its strict errors as correct. Under the judge's own verdict alone it ranks 7th, not 1st — a real, disclosed dependency on which of two defensible regimes is used, not evidence either regime is "correct."

> **Over a third of the benchmark's items carry no discriminative signal.** 148 of 406 items (36.5%) are answered identically by all twelve models and separate no pair of systems; a further 72 items are near-unanimous. Nominal item count and discriminative coverage are different quantities that benchmarks rarely distinguish — a reported accuracy gap is only as trustworthy as the fraction of items actually separating the systems being compared.

> **No model significantly beats a careful non-specialist human on the benchmark's harder half.** On the 60 shared medium-and-hard items the human scores 80.0% (48/60); the best model reaches 85.0% (Gemini 2.5 Flash, paired bootstrap p = 0.44, not significant). No model differs significantly from the human reader in either direction once corrected for the twelve-model family (Bonferroni α = 0.05/12).

> **Scale doesn't explain these results.** Llama 4 Scout 17B and LLaMA-3.3-70B are statistically indistinguishable (p = 0.786) across a four-fold parameter difference. GPT-OSS 120B and 20B are likewise indistinguishable (p = 0.912) across a six-fold difference within one model family — and their error overlap (Jaccard 0.427) is the highest of any pair in the study: they don't just score alike, they fail on substantially the same items.

> **Task-type performance is highly dissociated.** Numerical reasoning is the most discriminating task (35.9-point spread, 84.8% to 48.9%); contradiction detection has the highest mean (91.5%) but discriminates least once corrected for its class imbalance (85.5% majority-class baseline). Gemini 2.5 Pro ranks near the top on REG (89.7%) but last on NUM (48.9%) — aggregate accuracy misrepresents deployment suitability for a specific regulatory task.

---

## Live Demo

**→ [huggingface.co/spaces/Rajveer-code/IndiaFinBench](https://huggingface.co/spaces/Rajveer-code/IndiaFinBench)**

A production Flask application deployed on HuggingFace Spaces (Docker, free tier). Built without UI frameworks — vanilla JS, raw WebGL for the archive scene, custom GLSL shaders for the scroll-linked 3D document formation.

| Feature | Description |
|---|---|
| **Interactive Leaderboard** | Sortable table of 12 LLMs with 95% Wilson CIs and per-task breakdown (REG / NUM / CON / TMP) |
| **Performance Charts** | Animated bar chart with task-type tabs and human baseline reference |
| **Difficulty Analysis** | Performance breakdown by Easy / Medium / Hard with per-model drill-down |
| **Dataset Explorer** | Browse benchmark items filtered by task type and difficulty |
| **Live RAG Query** | Real-time hybrid retrieval over 192 regulatory documents via Groq LLaMA-3.3-70B |
| **Model Submission** | Submits a pre-filled GitHub issue with the exact evaluation command |

**Stack:** Python 3.11 · Flask 3 · Gunicorn · FAISS-CPU · sentence-transformers (BAAI/bge-base-en-v1.5) · rank-bm25 · Groq API · SQLite · Docker · Vanilla JS · Raw WebGL/GLSL

---

## Leaderboard

Zero-shot, closed-book evaluation on the full 406-item benchmark. All prompts provide only the relevant regulatory passage; no retrieval, no external context.

| Rank | Model | REG | NUM | CON | TMP | Overall | 95% CI |
|------|-------|:---:|:---:|:---:|:---:|:-------:|--------|
| 1 | **Gemini 2.5 Flash** | 93.1% | **84.8%** | 88.7% | 88.5% | **89.7%** | [86.3%, 92.3%] |
| 2 | Qwen3-32B | 85.1% | 77.2% | 90.3% | **92.3%** | 85.5% | [81.7%, 88.6%] |
| 3 | LLaMA-3.3-70B | 86.2% | 75.0% | 95.2% | 79.5% | 83.7% | [79.8%, 87.0%] |
| 4 | Llama 4 Scout 17B | 86.2% | 66.3% | **98.4%** | 84.6% | 83.3% | [79.3%, 86.6%] |
| 5 | Kimi K2 | **89.1%** | 65.2% | 91.9% | 75.6% | 81.5% | [77.5%, 85.0%] |
| 6 | Gemma 3 4B | 86.2% | 70.7% | 79.0% | 71.8% | 78.8% | [74.6%, 82.5%] |
| 7 | LLaMA-3-8B | 79.9% | 64.1% | 93.5% | 78.2% | 78.1% | [73.8%, 81.8%] |
| 8 | GPT-OSS 120B | 79.9% | 59.8% | 95.2% | 76.9% | 77.1% | [72.8%, 80.9%] |
| 9 | GPT-OSS 20B | 79.9% | 58.7% | 95.2% | 76.9% | 76.8% | [72.5%, 80.7%] |
| 10 | Gemini 2.5 Pro | 89.7% | 48.9% | 93.5% | 64.1% | 76.1% | [71.7%, 80.0%] |
| 11 | Mistral-7B | 79.9% | 66.3% | 80.6% | 74.4% | 75.9% | [71.5%, 79.8%] |
| 12 | DeepSeek-R1-Distill-Llama-70B | 72.4% | 69.6% | **96.8%** | 70.5% | 75.1% | [70.7%, 79.1%] |
| — | **Human (non-specialist)** *(n=60, med+hard)* | 100.0 | 56.2 | 82.4 | 87.5 | 80.0% | [68.2%, 88.2%] |

Strict string-matching pipeline, 95% Wilson score confidence intervals. Paired bootstrap significance (100,000 resamples) across all 66 model pairs finds 36 significantly different at p < 0.05, of which 15–16 survive Bonferroni correction (α = 0.05/66). This is one of three scoring regimes reported in the paper, not the sole ranking — see [Key Findings](#key-findings) above. Full significance matrix: `evaluation/bootstrap_significance_results.json`.

> **†** Claude 3 Haiku was evaluated on the initial 150-item development subset: Overall **91.3%**. Not directly comparable to the 406-item results above.

---

## Dataset and Task Taxonomy

```
IndiaFinBench  (406 items, sourced from 192 SEBI and RBI documents, 1992–2026)
│
├── REG — Regulatory Interpretation   174 items  (42.9%)
│         Given a regulatory passage, identify the applicable rule, threshold,
│         or scope. Tests precise reading of regulatory language.
│
├── NUM — Numerical Reasoning          92 items  (22.7%)
│         Compute results over figures embedded in regulatory text:
│         capital ratios, dividend limits, margin requirements, penalty calculations.
│
├── CON — Contradiction Detection      62 items  (15.3%)
│         Given two regulatory passages, determine whether they contradict each
│         other on a stated issue (Yes/No with justification required).
│
└── TMP — Temporal Reasoning           78 items  (19.2%)
          Establish the chronological ordering of regulatory events, identify
          which circular was operative at a given date, or compute elapsed time
          between regulatory milestones.
```

**Difficulty distribution:** Easy 160 (39.4%) · Medium 182 (44.8%) · Hard 64 (15.8%)

**Source documents:** 92 SEBI circulars + 100 RBI master directions and circulars. Full metadata in `data/metadata_sebi.csv` and `data/metadata_rbi.csv`.

**Why Indian regulatory text is uniquely challenging:**
- Numerical thresholds are embedded in dense regulatory prose, requiring multi-clause arithmetic
- Regulatory chains: a 2024 circular may supersede a 2019 one which itself amended a 2013 gazette — models must reason over this temporal web
- Jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA, SARFAESI) that models trained predominantly on Western corpora may not reliably interpret
- Contradiction detection requires holding two regulatory documents simultaneously in context and identifying logical conflicts on a specific issue

---

## Annotation Methodology

### Inter-Annotator Agreement

180 items were independently annotated across three rounds (44.3% benchmark coverage):

| Task | Items | Agreement | Cohen's κ |
|------|------:|:---------:|:---------:|
| Regulatory Interpretation | 63 | 92.1% | — |
| Numerical Reasoning | 44 | 81.8% | — |
| Contradiction Detection | 35 | 80.0% | **0.712** |
| Temporal Reasoning | 38 | 86.8% | — |
| **Overall** | **180** | **86.1%** | — |

κ = 0.712 for contradiction detection falls in the "substantial agreement" range (Landis & Koch, 1977). NUM and CON have the two lowest task-level agreement rates, both driven by the same mechanism: reference answers are terse final values, while the second annotator's answers often embed a full derivation the strict scorer's exact/numeric-match stages can miss. All eight discordant numerical-reasoning items were manually re-verified: every case's computed value was equivalent, disagreements were purely presentational. Full IAA data: `annotation/iaa/`.

### Model-Based Validation

LLaMA-3.3-70B independently attempted 150 items to verify unambiguous answerability from context. Overall agreement: **90.7%**. Cohen's κ = 0.918 for contradiction detection.

---

## Scoring Methodology

Answers are scored using a four-stage procedure applied in sequence:

1. **Exact match** — case-normalised and punctuation-stripped comparison
2. **Fuzzy token match** — RapidFuzz `token_set_ratio ≥ 0.72`
3. **Numerical extraction match** — handles currency symbols, commas, units (₹, lakh, crore, %)
4. **Yes/No match** — for contradiction detection items

The 0.72 fuzzy threshold was calibrated by manual inspection and validated against adjacent thresholds (0.65 too permissive, 0.80 too strict). Full ablation: `evaluation/error_analysis/fuzzy_ablation_*.csv`.

---

## Hybrid RAG System

A production-grade retrieval-augmented generation system for open-book querying of the full regulatory corpus — the open-book counterpart to the closed-book benchmark.

**Pipeline:**

```
Query → BGE Embedder ─→ FAISS index (dense, 768-dim, 4347 vectors) ─┐
                    └→ BM25 index  (sparse, rank-bm25)              ─┤ RRF (k=60) → Top-K chunks → Groq LLaMA-3.3-70B → Answer
                                                                      ┘
```

**Retrieval Ablation (6 configurations):**

| Config | Recall@5 | MRR | p50 latency |
|--------|:--------:|:---:|:-----------:|
| Dense only (B0) | 0.688 | 0.542 | 48 ms |
| BM25 only (B1) | 0.764 | **0.674** | 30 ms |
| **Hybrid RRF (B2)** ◄ selected | **0.785** | 0.640 | 77 ms |
| Small chunks 800-char (B3) | 0.583 | 0.493 | 138 ms |
| Large chunks 2400-char (B4) | 0.542 | 0.410 | 71 ms |
| Hybrid k=10 (B5) | 0.785 | 0.640 | 78 ms |

**Findings:** Hybrid RRF improves Recall@5 by +9.7pp over dense-only. BM25 achieves the best MRR, confirming that citation-heavy regulatory text with structured identifiers (circular numbers, section references) strongly favours lexical matching. 1,600-character chunking is the empirical optimum: smaller chunks fragment multi-clause provisions; larger chunks introduce retrieval noise.

**Embeddings:** BAAI/bge-base-en-v1.5 (768-dim) · **Index size:** FAISS 17 MB · BM25 18 MB · **Generator:** Groq `llama-3.3-70b-versatile`

```bash
# Build the index (~3 min on CPU)
python -m rag.scripts.build_index

# Run the 6-configuration retrieval ablation
python -m rag.scripts.run_evaluation
```

---

## Quick Start

### Load the Dataset

```python
from datasets import load_dataset

ds = load_dataset("Rajveer-code/IndiaFinBench", split="train")
print(f"Total items: {len(ds)}")   # 406

# Filter by task type
reg_items = ds.filter(lambda x: x["task_type"] == "regulatory_interpretation")
num_items = ds.filter(lambda x: x["task_type"] == "numerical_reasoning")
```

### Evaluate a New Model

```bash
# API model (OpenAI-compatible)
python evaluation/evaluate.py \
    --dataset data/benchmark/indiafinbench_v1.csv \
    --model gemini-2.5-flash \
    --provider google \
    --output results/predictions/gemini_flash.csv

# Local model via Ollama
python evaluation/evaluate.py \
    --dataset data/benchmark/indiafinbench_v1.csv \
    --model llama3:8b \
    --provider ollama \
    --output results/predictions/llama3_8b.csv
```

### Run the Demo Locally

```bash
git clone https://github.com/Rajveer-code/IndiaFinBench.git
cd IndiaFinBench
pip install -r demo/requirements.txt -r rag/requirements.txt

export GROQ_API_KEY="your_key_here"   # Free at console.groq.com
python demo/app.py
# → http://localhost:7860
```

### Regenerate All Figures and Statistics

```bash
# All paper figures + bootstrap / Wilson CI / difficulty analyses
python scripts/generate_figures.py
```

---

## Demo Application

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Research narrative + leaderboard page |
| `GET` | `/api/leaderboard` | JSON — 12 models + human baseline with CIs |
| `GET` | `/api/example?task=&diff=` | Random benchmark item (filterable by task and difficulty) |
| `POST` | `/api/rag` | Hybrid RAG query (rate-limited 20 req/min) |
| `POST` | `/api/submit` | Returns pre-filled GitHub issue URL for model submission |

### Deployment Architecture

```
HuggingFace Spaces  (Docker, CPU basic, free — 2 vCPU / 16 GB RAM)
│
├── Gunicorn  (1 worker, 4 threads, port 7860)
│   └── Flask  demo/app.py
│
├── RAG pipeline  rag/
│   ├── BGE embedder  (baked into Docker image at build time, ~270 MB)
│   ├── FAISS index   rag/index/faiss.index  (17 MB, stored via Git LFS)
│   └── BM25 index    rag/index/bm25.pkl     (18 MB, stored via Git LFS)
│
└── SQLite  demo/leaderboard.db  (seeded at startup from baselines.json)
```

Redeploy to HuggingFace Spaces:

```bash
bash scripts/deploy_space.sh
```

---

## Repository Structure

```
IndiaFinBench/
│
├── data/
│   ├── benchmark/indiafinbench_v1.csv     # Canonical 406-item benchmark
│   ├── metadata_sebi.csv                  # 92 SEBI source documents with URLs
│   └── metadata_rbi.csv                   # 100 RBI source documents with URLs
│
├── annotation/
│   ├── raw_qa/                            # Full benchmark JSON (406 + 150-item dev subset)
│   ├── guidelines/annotation_guide_v1.md  # Annotation protocol and decision rules
│   ├── iaa/                               # Inter-annotator agreement data (180 items, 3 rounds)
│   └── human_eval/                        # Human evaluation responses (n=60, single non-specialist)
│
├── evaluation/
│   ├── evaluate.py                        # Canonical evaluation entry point
│   ├── prompts/                           # Per-task-type system prompts
│   ├── results/                           # Per-model prediction CSVs (12 models)
│   ├── error_analysis/                    # Error taxonomy, bootstrap matrix, fuzzy ablation
│   └── novel_methods/                     # 11 novel methodological analyses
│
├── results/
│   ├── predictions/                       # Canonical predictions for all 12 models
│   └── aggregate/all_model_results.csv    # Aggregated results table
│
├── scripts/
│   ├── generate_figures.py                # All paper figures and statistical outputs
│   ├── bootstrap_significance.py          # Paired bootstrap (10,000 resamples)
│   ├── wilson_ci.py                       # 95% Wilson CI computation
│   ├── compute_kappa.py                   # Inter-annotator Cohen's kappa
│   ├── deploy_space.sh                    # Filtered LFS deploy to HuggingFace Spaces
│   └── exp[1-11]_*.py                     # Novel methodological analysis scripts
│
├── rag/                                   # Hybrid RAG pipeline
│   ├── pipeline.py                        # RAGPipeline orchestrator
│   ├── embeddings.py                      # BGE embedder (asymmetric query/corpus)
│   ├── index.py                           # FAISS dense index
│   ├── bm25_index.py                      # BM25 sparse index
│   ├── retriever.py                       # HybridRetriever with RRF fusion
│   ├── generator.py                       # Groq LLM generation
│   ├── config.py                          # RAGConfig dataclass
│   └── index/
│       ├── faiss.index                    # 17 MB FAISS flat index (Git LFS)
│       ├── bm25.pkl                       # 18 MB BM25 serialised model (Git LFS)
│       └── chunks.pkl                     # 9.8 MB chunk metadata (Git LFS)
│
├── demo/                                  # Live web application
│   ├── app.py                             # Flask app (leaderboard, RAG, submit APIs)
│   ├── requirements.txt
│   ├── templates/index.html               # Seven-chapter scroll narrative (raw WebGL archive scene)
│   ├── static/css/main.css                # Archival-editorial design system
│   ├── static/js/
│   │   ├── archive-scene.js               # Raw WebGL + GLSL: 192-card 3D formation with scroll morphing
│   │   ├── data.js                        # Model data + Wilson CI bounds
│   │   └── main.js                        # Charts, tables, RAG UI, dataset explorer, submit
│   ├── database/db.py                     # SQLite leaderboard (init + query)
│   ├── data/
│   │   ├── questions.json                 # 406 benchmark items (dataset explorer)
│   │   └── baselines.json                 # Baseline model results (seeds DB at startup)
│   └── tests/test_app.py                  # 14 API behaviour tests
│
├── paper/
│   ├── tmlr/                              # Current manuscript, target venue TMLR
│   │   ├── draft_*.tex                    # Section sources, assembled by tmlr_submission/main.tex
│   │   └── tmlr_submission/main.pdf       # Build output (gitignored; run latexmk to produce)
│   └── figures/                           # Publication figures
│
├── Dockerfile                             # Root Dockerfile for HuggingFace Spaces
├── .dockerignore
├── README.md
└── LICENSE
```

---

## Citation

```bibtex
@misc{pall2026indiafinbench,
  title         = {{IndiaFinBench}: An Evaluation Benchmark for Large Language Model Performance
                   on Indian Financial Regulatory Text},
  author        = {Pall, Rajveer Singh},
  year          = {2026},
  eprint        = {2604.19298},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2604.19298}
}
```

---

## License

| Component | License |
|-----------|---------|
| Dataset (`data/benchmark/`, `annotation/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution |
| Code (`scripts/`, `evaluation/`, `demo/`, `rag/`) | [MIT License](LICENSE) |
| Source regulatory documents | Public domain (Government of India) |

---

## Contact

**Rajveer Singh Pall** — [rajveerpall04@gmail.com](mailto:rajveerpall04@gmail.com)

For questions about the benchmark methodology, to report annotation issues, or for collaboration inquiries, please open an issue or reach out directly.
