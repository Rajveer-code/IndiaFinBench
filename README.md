<div align="center">

# IndiaFinBench

### The First Evaluation Benchmark for Large Language Models on Indian Financial Regulatory Text

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-HuggingFace_Spaces-FFD21E?style=flat-square)](https://huggingface.co/spaces/Rajveer-code/IndiaFinBench)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-Rajveer--code%2FIndiaFinBench-FFD21E?style=flat-square)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue?style=flat-square)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2604.19298-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.19298)
[![Under review: FinNLP @ EMNLP 2026](https://img.shields.io/badge/Under_review-FinNLP_%40_EMNLP_2026-red?style=flat-square)](https://arxiv.org/abs/2604.19298)

<br>

| 406 | 192 | 12 | 0.785 | 80.0% |
|:---:|:---:|:---:|:---:|:---:|
| Expert-annotated QA items | SEBI + RBI documents | LLMs benchmarked | Hybrid RAG Recall@5 | Human reference (n=60, med+hard) |

</div>

---

## Overview

Existing financial NLP benchmarks — FinQA, ConvFinQA, FLUE — evaluate models on Western markets using SEC filings and US news. No public benchmark tests LLM performance on Indian regulatory text, despite India having one of the world's most prolific regulatory document ecosystems (SEBI, RBI, IRDAI, PFRDA) with millions of participants governed by its rules.

IndiaFinBench fills this gap. It is a **zero-shot, closed-book evaluation benchmark** of 406 expert-annotated question-answer pairs drawn from 192 circulars, regulations, and master directions published by SEBI and the RBI between 1992 and 2026. The benchmark tests four distinct reasoning capabilities — regulatory interpretation, numerical computation, contradiction detection across circulars, and temporal reasoning over chronological document sequences — each of which must be solved from document evidence alone, without retrieval.

The benchmark reveals that **strict string-matching and semantic (LLM-judge-corrected) scoring tell substantively different stories** — the two regimes bracket true accuracy and separate genuine capability gaps from format non-compliance — that efficiency dominates scale (a 17B-parameter model statistically matches a 70B model), and that on a 60-item medium-and-hard subset no model significantly outperforms a careful non-specialist human (80.0%), while the weakest model is significantly worse.

---

## Research Contributions

1. **First Indian financial regulatory NLP benchmark** — 406 items across four task types, sourced from 192 SEBI and RBI documents spanning 34 years (1992–2026). Fills a documented gap in non-Western financial NLP evaluation.

2. **Four-axis task taxonomy** — REG (regulatory interpretation), NUM (numerical reasoning), CON (contradiction detection), TMP (temporal reasoning) — designed to expose orthogonal capabilities that aggregate accuracy conceals.

3. **Human reference point with inter-annotator agreement** — 60-item human evaluation by a single non-specialist evaluator on medium and hard items (80.0%, 95% Wilson CI [68.2%, 88.2%]) with paired human-vs-model bootstrap tests, plus a 180-item three-round IAA study. No model significantly outperforms the human on the shared items; only the weakest is significantly worse.

4. **Bootstrap statistical significance at scale** — Paired bootstrap (10,000 resamples) across all 66 model pairs under both scoring regimes (42/66 significant at p < 0.05 strict; 19/66 after Bonferroni), resolving which leaderboard gaps reflect genuine capability differences.

5. **Efficiency-over-scale finding** — Llama 4 Scout 17B matches LLaMA-3.3-70B (p = 0.80) with one-quarter the parameters. GPT-OSS 120B and 20B are statistically indistinguishable (p = 0.86). Both null results persist under judge-corrected scoring, directly challenging the scaling assumption in regulatory NLP deployment decisions.

6. **Hybrid RAG system with full ablation** — Production-grade FAISS + BM25 pipeline with Reciprocal Rank Fusion, benchmarked across six retrieval configurations. Hybrid RRF improves Recall@5 by +9.7pp over dense-only retrieval; optimal chunk size empirically determined at 1,600 characters.

7. **Open, reproducible artefacts** — Dataset on HuggingFace, evaluation code, annotation guidelines, prediction CSVs for all 12 models, statistical scripts, and a live deployed demo — all in this repository.

---

## Key Findings

> **No model significantly beats a careful non-specialist human on the benchmark's harder half.** On the 60 shared medium-and-hard items the human scores 80.0%; the best model reaches 85.0% (paired bootstrap p = 0.48, not significant) and only Gemma 4 E4B is significantly worse (63.3%, p = 0.011). Models decisively exceed the human only on multi-step numerical reasoning (best model 84.8% vs human 56.2%).

> **NUM is the most discriminating task type.** A 35.9 percentage-point spread between best (Gemini 2.5 Flash: 84.8%) and worst (Gemini 2.5 Pro: 48.9%) on numerical reasoning — versus only ~13pp on regulatory interpretation — identifies NUM as the primary capability differentiator.

> **DeepSeek R1 extractability gap.** The reasoning-specialised DeepSeek R1 70B ranks 11th under strict scoring, yet the item-level judge audit reclassifies 86% of its 101 errors as format non-compliance; its judge-corrected accuracy (96.6%) is statistically tied with the top cluster. Reasoning-style output fails deployment-style extractive pipelines — a capability-vs-extractable-capability distinction that single-metric evaluation conflates.

> **Three descriptive performance clusters under strict scoring.** Tier 1 = Gemini 2.5 Flash, Qwen3-32B, LLaMA-3.3-70B, Llama 4 Scout 17B, Kimi K2 (81.5–89.7%); Tier 2 = LLaMA-3-8B through DeepSeek R1 70B (75–79%); Tier 3 = Gemma 4 E4B (70.4%). The top-versus-bottom separation is Bonferroni-robust; the middle boundaries are graded, not categorical (e.g., Gemma vs Mistral p = 0.072). Under judge-corrected scoring, 11 of 12 models converge to 90.6–96.6% and only Gemma remains separable.

> **Task-type performance is highly dissociated.** Gemini 2.5 Pro ranks 1st on REG (89.7%) but last on NUM (48.9%) within the same model. Aggregate accuracy misrepresents deployment suitability for specific regulatory tasks.

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
| 6 | LLaMA-3-8B | 79.9% | 64.1% | 93.5% | 78.2% | 78.1% | [73.8%, 81.8%] |
| 7 | GPT-OSS 120B | 79.9% | 59.8% | 95.2% | 76.9% | 77.1% | [72.8%, 80.9%] |
| 8 | GPT-OSS 20B | 79.9% | 58.7% | 95.2% | 76.9% | 76.8% | [72.5%, 80.7%] |
| 9 | Gemini 2.5 Pro | 89.7% | 48.9% | 93.5% | 64.1% | 76.1% | [71.8%, 80.0%] |
| 10 | Mistral-7B | 79.9% | 66.3% | 80.6% | 74.4% | 75.9% | [71.5%, 79.8%] |
| 11 | DeepSeek R1 70B | 72.4% | 69.6% | **96.8%** | 70.5% | 75.1% | [70.7%, 79.1%] |
| 12 | Gemma 4 E4B | 83.9% | 50.0% | 72.6% | 62.8% | 70.4% | [65.8%, 74.7%] |
| — | **Human (non-specialist)** *(n=60, med+hard)* | 100.0 | 56.2 | 82.4 | 87.5 | 80.0% | [68.2%, 88.2%] |

95% Wilson score confidence intervals. Paired bootstrap significance (10,000 resamples) across all 66 model pairs confirms three statistically distinct performance tiers. Full significance matrix: `evaluation/bootstrap_significance_results.json`.

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
| Regulatory Interpretation | 63 | 85.7% | — |
| Numerical Reasoning | 44 | 59.1% | — |
| Contradiction Detection | 35 | 88.6% | **0.645** |
| Temporal Reasoning | 38 | 73.7% | — |
| **Overall** | **180** | **77.2%** | — |

κ = 0.645 for contradiction detection falls in the "substantial agreement" range (Landis & Koch, 1977). NUM agreement of 59.1% reflects a formatting artefact: reviewer notes included derivations where reference answers give concise final values. Post-hoc review of all 26 NUM disagreements confirmed zero substantive arithmetic errors. Full IAA data: `annotation/iaa/`.

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
│   ├── indiafinbench_paper_v12.md         # Current paper draft (target: EMNLP 2026)
│   ├── references.bib
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
