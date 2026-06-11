# IndiaFinBench

**The first publicly available evaluation benchmark for large language model performance on Indian financial regulatory text.**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-HuggingFace_Spaces-FFD21E?style=flat)](https://huggingface.co/spaces/Rajveer-code/IndiaFinBench)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-Rajveer--code%2FIndiaFinBench-FFD21E)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Target: EMNLP 2026](https://img.shields.io/badge/Target-EMNLP%202026-red.svg)]()

> **406 expert-annotated QA items · 192 SEBI + RBI documents · 12 LLMs evaluated · Hybrid RAG with Recall@5 = 0.785**

---

## 🌐 Live Demo

**→ [huggingface.co/spaces/Rajveer-code/IndiaFinBench](https://huggingface.co/spaces/Rajveer-code/IndiaFinBench)**

A fully working Flask web application deployed on HuggingFace Spaces (Docker, CPU, free tier):

| Feature | Description |
|---|---|
| **Interactive Leaderboard** | Sortable table of 12 LLMs with 95% Wilson CIs, per-task breakdown (REG / NUM / CON / TMP), difficulty analysis |
| **Animated Bar Charts** | Hover-interactive visualisations of task and difficulty performance |
| **Dataset Explorer** | Browse random benchmark items filtered by task type and difficulty |
| **RAG Demo** | Live hybrid retrieval (FAISS + BM25 + RRF) over 192 regulatory documents, answered by Groq LLaMA-3.3-70B |
| **Model Submission** | Opens a pre-filled GitHub issue with the exact evaluation command |

**Stack:** Python 3.11 · Flask 3 · Gunicorn · FAISS-CPU · sentence-transformers (BAAI/bge-base-en-v1.5) · rank-bm25 · Groq API · SQLite · Docker

---

## What is IndiaFinBench?

IndiaFinBench is a zero-shot evaluation benchmark consisting of **406 expert-annotated question-answer pairs** drawn from **192 documents** published by the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning 1992–2026. It tests whether large language models can reliably reason about Indian financial regulatory text — a domain with distinctive challenges not captured by existing Western-centric financial NLP benchmarks.

Indian regulatory documents embed numerical thresholds in dense prose, reference chains of superseding circulars that require temporal reasoning to untangle, and use jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA) that models trained on Western corpora may not reliably interpret. IndiaFinBench makes these challenges measurable.

---

## Leaderboard

Results on the full 406-item benchmark under zero-shot, context-only evaluation:

| Rank | Model | REG | NUM | CON | TMP | Overall | 95% CI |
|------|-------|-----|-----|-----|-----|---------|--------|
| 1 | **Gemini 2.5 Flash** | 93.1% | 84.8% | 88.7% | 88.5% | **89.7%** | [86.3%, 92.3%] |
| 2 | Qwen3-32B | 85.1% | 77.2% | 90.3% | **92.3%** | 85.5% | [81.7%, 88.6%] |
| 3 | LLaMA-3.3-70B | 86.2% | 75.0% | 95.2% | 79.5% | 83.7% | [79.8%, 87.0%] |
| 4 | Llama 4 Scout 17B | 86.2% | 66.3% | **98.4%** | 84.6% | 83.3% | [79.3%, 86.6%] |
| 5 | Kimi K2 | **89.1%** | 65.2% | 91.9% | 75.6% | 81.5% | [77.5%, 85.0%] |
| 6 | LLaMA-3-8B | 79.9% | 64.1% | 93.5% | 78.2% | 78.1% | [73.8%, 81.8%] |
| 7 | GPT-OSS 120B | 79.9% | 59.8% | 95.2% | 76.9% | 77.1% | [72.8%, 80.9%] |
| 8 | GPT-OSS 20B | 79.9% | 58.7% | 95.2% | 76.9% | 76.8% | [72.5%, 80.7%] |
| 9 | Gemini 2.5 Pro | **89.7%** | 48.9% | 93.5% | 64.1% | 76.1% | [71.8%, 80.0%] |
| 10 | Mistral-7B | 79.9% | 66.3% | 80.6% | 74.4% | 75.9% | [71.5%, 79.8%] |
| 11 | DeepSeek R1 70B | 72.4% | 69.6% | **96.8%** | 70.5% | 75.1% | [70.7%, 79.1%] |
| 12 | Gemma 4 E4B | 83.9% | 50.0% | 72.6% | 62.8% | 70.4% | [65.8%, 74.7%] |
| — | Human Expert (n=100) | — | — | — | — | 69.0% | [59.4%, 77.2%] |

> **†** Claude 3 Haiku was evaluated on the initial 150-item subset (REG=53, NUM=32, CON=30, TMP=35): Overall **91.3%**. Provided for contextualisation; not directly comparable to the 406-item results above.

95% Wilson score confidence intervals. Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs confirms **three statistically distinct performance tiers**. Full significance matrix in `evaluation/bootstrap_significance_results.json`.

---

## Task Types

```
IndiaFinBench (406 items)
├── REG — Regulatory Interpretation  (174 items, 42.9%)
│         Given a regulatory passage, identify the correct rule, threshold,
│         or scope of applicability. Tests precision reading of regulatory language.
│
├── NUM — Numerical Reasoning         (92 items, 22.7%)
│         Perform arithmetic over figures embedded in regulatory text:
│         capital ratios, dividend limits, margin requirements.
│
├── CON — Contradiction Detection     (62 items, 15.3%)
│         Given two regulatory passages, determine whether they contradict
│         each other on the stated issue (Yes/No + explanation required).
│
└── TMP — Temporal Reasoning          (78 items, 19.2%)
          Establish the chronological ordering of regulatory events, identify
          which circular was operative at a given time, or compute elapsed time
          between regulatory milestones.
```

**Difficulty:** Easy 160 (39.4%) · Medium 182 (44.8%) · Hard 64 (15.8%)

---

## Key Findings

- **Three performance tiers confirmed by bootstrap testing:** Frontier API models (81–90%), mid-tier open-weight models (75–79%), and Gemma 4 E4B (70%). Most cross-tier differences are statistically significant (p < 0.05).
- **Efficiency over scale:** Llama 4 Scout 17B statistically matches LLaMA-3.3-70B (p = 0.79) with one-quarter the parameters — strong evidence that scale alone does not drive regulatory reasoning.
- **Scaling plateau:** GPT-OSS 120B and GPT-OSS 20B are statistically indistinguishable (p = 0.91, Δ = +0.3pp).
- **DeepSeek R1 paradox:** Despite being reasoning-specialised, DeepSeek R1 70B ranks 11th — particularly weak on temporal reasoning (70.5%), exposing a gap between general chain-of-thought capability and domain-specific timeline reasoning.
- **Gemini 2.5 Pro NUM dissociation:** Scores only 48.9% on NUM (lowest of any model) while ranking 1st on REG (89.7%), demonstrating that task-type performance can be highly dissociated even within the same model.
- **NUM as the key discriminator:** 35.9pp spread between best (Gemini 2.5 Flash: 84.8%) and worst (Gemini 2.5 Pro: 48.9%) — the most informative task type for differentiating model capability.
- **All 12 models beat the human baseline:** Human expert accuracy = 69.0% (n=100, 95% CI [59.4%, 77.2%]); all 12 evaluated LLMs exceed this, with Gemini 2.5 Flash leading at 89.7%.

---

## Hybrid RAG System

This repository also includes a hybrid retrieval-augmented generation (RAG) system for **open-book querying** of the regulatory corpus — the open-book counterpart to the closed-book benchmark above. It is integrated into the live demo.

**Pipeline:**

```
Query → BGE Embedder → FAISS (dense)  ┐
                   → BM25  (sparse) ──┤ RRF → Top-K chunks → Groq LLaMA-3.3-70B → Answer
                                       ┘
```

- **Embeddings:** BAAI/bge-base-en-v1.5 (768-dim, asymmetric query/corpus encoding)
- **Sparse index:** BM25 (rank-bm25) over 1600-character chunks
- **Dense index:** FAISS flat L2 (17 MB, 4,347 vectors)
- **Fusion:** Reciprocal Rank Fusion (k = 60)
- **Generator:** Groq `llama-3.3-70b-versatile` (14,400 free requests/day)

### Retrieval Ablation Results

| Config | Recall@5 | MRR | p50 ms |
|--------|----------|-----|--------|
| Dense only (B0) | 0.688 | 0.542 | 48 |
| BM25 only (B1) | 0.764 | **0.674** | 30 |
| **Hybrid RRF (B2)** ◄ | **0.785** | 0.640 | 77 |
| Small chunks 800-char (B3) | 0.583 | 0.493 | 138 |
| Large chunks 2400-char (B4) | 0.542 | 0.410 | 71 |
| Hybrid k=10 (B5) | 0.785 | 0.640 | 78 |

Hybrid RRF improves Recall@5 by **+9.7pp** over dense-only. BM25 achieves the best MRR, confirming that citation-heavy regulatory text with structured identifiers strongly favours lexical matching. 1600-char chunking is the empirical optimum: smaller chunks fragment multi-clause provisions; larger chunks introduce retrieval noise.

```bash
# Build the index from the parsed corpus (~3 min on CPU)
python -m rag.scripts.build_index

# Run the 6-configuration retrieval ablation
python -m rag.scripts.run_evaluation
```

---

## Demo Application

### Running Locally

```bash
# Clone and install
git clone https://github.com/Rajveer-code/IndiaFinBench.git
cd IndiaFinBench
pip install -r demo/requirements.txt -r rag/requirements.txt

# Set Groq API key (free at console.groq.com)
export GROQ_API_KEY="your_key_here"

# Start the server
python demo/app.py
# → http://localhost:7860
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Leaderboard HTML page |
| `GET` | `/api/leaderboard` | JSON — 12 models + human baseline |
| `GET` | `/api/example?task=&diff=` | Random benchmark item (filterable) |
| `POST` | `/api/rag` | Hybrid RAG query (rate-limited 20 req/min) |
| `POST` | `/api/submit` | Returns pre-filled GitHub issue URL |

### Deployment Architecture

```
HuggingFace Spaces (Docker, CPU basic, free — 2 vCPU / 16 GB RAM)
│
├── Gunicorn (1 worker, 4 threads, port 7860)
│   └── Flask app  demo/app.py
│
├── RAG pipeline  rag/
│   ├── BGE embedder (pre-baked in Docker image, ~270 MB)
│   ├── FAISS index  rag/index/faiss.index  (17 MB)
│   └── BM25  index  rag/index/bm25.pkl     (18 MB)
│
└── SQLite  demo/leaderboard.db  (seeded at startup, read-only at runtime)
```

The BGE model is downloaded once at Docker build time and baked into the image, so the first RAG request is fast (~200 ms) rather than triggering a 30-second cold start.

**Deploying:** the Space repo is a filtered artifact of this repository — only the files the Docker build needs, with index binaries stored via Git LFS (a HuggingFace requirement). Redeploy with:

```bash
bash scripts/deploy_space.sh
```

### Running Tests

```bash
pytest demo/tests/test_app.py -v
# 14 passed in <2s
```

---

## Quick Start — Benchmark Evaluation

### Load the Dataset

```python
from datasets import load_dataset

ds = load_dataset("Rajveer-code/IndiaFinBench", split="train")
print(f"Total items: {len(ds)}")  # 406

# Filter by task type
reg_items = ds.filter(lambda x: x["task_type"] == "regulatory_interpretation")
print(f"REG items: {len(reg_items)}")  # 174
```

### Evaluate a New Model

```bash
# API model
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

### Regenerate Figures and Statistics

```bash
# All paper figures + bootstrap / Wilson CI / difficulty outputs
python scripts/generate_figures.py
```

---

## Annotation Quality

**Model-based validation (150 items):** LLaMA-3.3-70B independently attempted each item to verify unambiguous answerability from context. Overall agreement: **90.7%**. Cohen's κ = 0.918 for contradiction detection (binary Yes/No).

**Human inter-annotator agreement (180 items, three annotation rounds, 44.3% benchmark coverage):**

| Task | Items | Agreement | Cohen's κ |
|------|-------|-----------|-----------|
| Regulatory Interpretation | 63 | 85.7% | — |
| Numerical Reasoning | 44 | 59.1% | — |
| Contradiction Detection | 35 | 88.6% | **0.645** |
| Temporal Reasoning | 38 | 73.7% | — |
| **Overall** | **180** | **77.2%** | — |

κ = 0.645 for contradiction detection falls in the "substantial agreement" range (Landis & Koch, 1977). NUM agreement of 59.1% reflects a formatting artefact: annotator2 provides verbose step-by-step derivations whereas reference answers give concise final values; post-hoc review of all 26 NUM disagreements confirmed zero substantive arithmetic errors. Full IAA data in `annotation/iaa/`.

---

## Scoring Methodology

Answers are scored using a **four-stage procedure**:

1. **Exact match** — after case-normalisation and punctuation stripping
2. **Fuzzy token match** — RapidFuzz `token_set_ratio ≥ 0.72`
3. **Numerical extraction match** — handles currency symbols, commas, units
4. **Yes/No match** — for contradiction detection items

The 0.72 fuzzy threshold was calibrated by manual inspection of borderline cases and validated against adjacent thresholds (0.65 too permissive, 0.80 too strict). Full ablation in `evaluation/error_analysis/fuzzy_ablation_*.csv`.

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
│   ├── raw_qa/                            # Full benchmark JSON (406 + 150-item subset)
│   ├── guidelines/annotation_guide_v1.md  # Annotation protocol
│   ├── iaa/                               # Inter-annotator agreement data (180 items, 3 rounds)
│   └── human_eval/                        # Human evaluation responses (n=100)
│
├── evaluation/
│   ├── evaluate.py                        # Canonical evaluation entry point
│   ├── prompts/                           # Per-task-type system prompts
│   ├── results/                           # Per-model prediction CSVs
│   ├── error_analysis/                    # Error taxonomy, bootstrap, Wilson CI
│   └── novel_methods/                     # 11 novel methodological analyses
│
├── results/
│   ├── predictions/                       # Canonical predictions (12 models)
│   └── aggregate/all_model_results.csv    # Aggregated Table 6 results
│
├── scripts/
│   ├── generate_figures.py                # All paper figures + statistical outputs
│   ├── bootstrap_significance.py          # Paired bootstrap (10k resamples)
│   ├── wilson_ci.py                       # 95% Wilson CI computation
│   ├── compute_kappa.py                   # IAA Cohen's kappa
│   ├── deploy_space.sh                    # Filtered LFS deploy to HF Spaces
│   └── exp[1-11]_*.py                     # Novel methodological analysis scripts
│
├── rag/                                   # Hybrid RAG pipeline
│   ├── pipeline.py                        # Top-level RAGPipeline orchestrator
│   ├── embeddings.py                      # BGE embedder (asymmetric query/corpus)
│   ├── index.py                           # FAISS dense index
│   ├── bm25_index.py                      # BM25 sparse index
│   ├── retriever.py                       # HybridRetriever with RRF fusion
│   ├── generator.py                       # Groq LLM generation
│   ├── config.py                          # RAGConfig dataclass
│   ├── index/                             # Pre-built indices
│   │   ├── faiss.index                    #   17 MB FAISS flat index
│   │   ├── bm25.pkl                       #   18 MB BM25 serialised model
│   │   └── chunks.pkl                     #   9.8 MB chunk metadata
│   └── scripts/
│       ├── build_index.py                 # Index construction script
│       └── run_evaluation.py              # 6-config ablation evaluation
│
├── demo/                                  # Live web application
│   ├── app.py                             # Flask app (leaderboard, RAG, submit APIs)
│   ├── requirements.txt                   # Flask, gunicorn, FAISS, sentence-transformers
│   ├── templates/index.html               # Single-page research site (7-chapter scroll narrative)
│   ├── static/css/main.css                # Design system (archival-editorial light theme)
│   ├── static/js/
│   │   ├── data.js                        # Model data + Wilson CI bounds (frontend)
│   │   └── main.js                        # Charts, tables, canvas, RAG UI, explorer, submit
│   ├── database/db.py                     # SQLite leaderboard (init + query)
│   ├── data/
│   │   ├── questions.json                 # 406 benchmark items (for dataset explorer)
│   │   └── baselines.json                 # Baseline model results (seeds DB)
│   └── tests/test_app.py                  # 14 API behaviour tests
│
├── paper/
│   ├── indiafinbench_paper_v12.md         # Current paper draft (EMNLP 2026)
│   ├── references.bib                     # BibTeX references
│   └── figures/                           # Publication figures
│
├── Dockerfile                             # Root Dockerfile for HF Spaces (Docker SDK)
├── .dockerignore                          # Excludes .env, corpus, research artefacts
├── README.md                              # This file
└── LICENSE
```

---

## Citation

```bibtex
@article{pall2026indiafinbench,
  title     = {{IndiaFinBench}: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text},
  author    = {Pall, Rajveer Singh},
  journal   = {Proceedings of EMNLP},
  year      = {2026},
  url       = {https://github.com/Rajveer-code/IndiaFinBench}
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

**Rajveer Singh Pall** — rajveerpall04@gmail.com

For questions about the benchmark, evaluation methodology, or collaboration inquiries, open an issue or contact directly.
