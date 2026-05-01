# IndiaFinBench

**The first publicly available evaluation benchmark for large language model performance on Indian financial regulatory text.**

[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![EMNLP 2026](https://img.shields.io/badge/Target-EMNLP%202026-red.svg)]()

> Benchmark + hybrid retrieval system for regulatory text, achieving Recall@5 = 0.785 with ablation across 6 configurations.

---

## What is IndiaFinBench?

IndiaFinBench is a zero-shot evaluation benchmark consisting of 406 expert-annotated question-answer pairs drawn from 192 documents published by the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI). It is designed to test whether large language models can reliably reason about Indian financial regulatory text — a domain with distinctive challenges not captured by existing Western-centric financial NLP benchmarks.

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
| — | Human Expert (n=30) | 55.6% | 44.4% | 83.3% | 66.7% | 60.0% | — |

| — | †Claude 3 Haiku | 92.5% | **93.8%** | 86.7% | 91.4% | 91.3% | [85.7%, 94.9%] |

> **†** Claude 3 Haiku was evaluated on the initial 150-item subset (REG=53, NUM=32, CON=30, TMP=35) due to API access constraints at the time of evaluation. Its result is provided for contextualisation but is not directly comparable to the 406-item results.

95% Wilson score confidence intervals. Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs confirms three statistically distinct performance tiers. Full significance results in `evaluation/bootstrap_significance_results.json`.

---

## Task Types

```
IndiaFinBench (406 items)
├── REG — Regulatory Interpretation (174 items, 42.9%)
│         Given a regulatory passage, identify the correct rule, threshold,
│         or scope of applicability. Tests precision reading of regulatory language.
│
├── NUM — Numerical Reasoning (92 items, 22.7%)
│         Perform arithmetic over figures embedded in regulatory text —
│         capital ratios, dividend limits, margin requirements.
│
├── CON — Contradiction Detection (62 items, 15.3%)
│         Given two regulatory passages, determine whether they contradict
│         each other on the stated issue (Yes/No + explanation).
│
└── TMP — Temporal Reasoning (78 items, 19.2%)
          Establish the chronological ordering of regulatory events, identify
          which circular was operative at a given time, or compute elapsed time
          between milestones.
```

Difficulty distribution: Easy 160 (39.4%) · Medium 182 (44.8%) · Hard 64 (15.8%)

---

## Quick Start

### Load from HuggingFace

```python
from datasets import load_dataset

ds = load_dataset("Rajveer-code/IndiaFinBench", split="train")
print(f"Total items: {len(ds)}")  # 406

# Filter by task type
reg_items = ds.filter(lambda x: x["task_type"] == "regulatory_interpretation")
print(f"REG items: {len(reg_items)}")  # 174

# Inspect a single item
item = ds[0]
print(item["context"])
print(item["question"])
print(item["reference_answer"])
```

### Run Evaluation on a New Model

```bash
# Clone the repository
git clone https://github.com/Rajveer-code/IndiaFinBench.git
cd IndiaFinBench

# Install dependencies
pip install -r requirements.txt

# Run zero-shot evaluation (API model example)
python evaluation/evaluate.py \
    --dataset data/benchmark/indiafinbench_v1.csv \
    --model gemini-2.5-flash \
    --provider google \
    --output results/predictions/my_model.csv

# Run evaluation on a local model via Ollama
python evaluation/evaluate.py \
    --dataset data/benchmark/indiafinbench_v1.csv \
    --model llama3:8b \
    --provider ollama \
    --output results/predictions/my_model.csv
```

### Regenerate All Figures and Statistical Outputs

```bash
# Generates all paper figures + bootstrap/Wilson CI/difficulty outputs
python scripts/generate_figures.py
```

This single script produces:
- `paper/figures/performance_heatmap.png`
- `paper/figures/radar_chart.png`
- `paper/figures/difficulty_lineplot.png`
- `paper/figures/inter_task_correlation.png`
- `evaluation/bootstrap_significance_results.json`
- `evaluation/wilson_ci_results.json`
- `evaluation/difficulty_breakdown.csv`
- `evaluation/task_accuracy_matrix.csv`

---

## Repository Structure

```
IndiaFinBench/
│
├── data/
│   ├── benchmark/
│   │   └── indiafinbench_v1.csv                 # Canonical 406-item benchmark (CSV)
│   ├── metadata_sebi.csv                        # 92 SEBI source documents with URLs
│   └── metadata_rbi.csv                         # 100 RBI source documents with URLs
│
├── annotation/
│   ├── raw_qa/
│   │   ├── indiafinbench_qa_combined_406.json   # Full benchmark (JSON, original format)
│   │   └── indiafinbench_qa_combined_150.json   # Initial 150-item subset
│   ├── guidelines/
│   │   └── annotation_guide_v1.md               # Annotation protocol
│   ├── iaa/
│   │   ├── iaa_60item_sample.csv                # Original 60-item IAA (κ=0.611 CON, 85% overall)
│   │   ├── iaa_60item_summary.csv               # IAA summary statistics by task type
│   │   ├── iaa_expansion_annotator2.csv         # Expansion 60-item IAA (annotator 2 responses)
│   │   └── iaa_expansion_reference.csv          # Expansion 60-item reference answers
│   └── human_eval/                              # Human evaluation responses (n=30)
│
├── evaluation/
│   ├── evaluate.py                              # Canonical evaluation entry point
│   ├── prompts/                                 # Task-type system prompts
│   ├── results/                                 # Per-model prediction CSVs (legacy location)
│   ├── results_judged/                          # LLM-as-judge re-scored predictions
│   ├── results_fewshot/                         # 3-shot evaluation results
│   ├── error_analysis/                          # Error taxonomy, bootstrap, Wilson CI
│   └── novel_methods/                           # 11 novel methodological analyses
│
├── results/
│   ├── predictions/                             # Canonical per-model predictions (12 models)
│   │   ├── gemini_2_5_flash.csv
│   │   ├── gemini_2_5_pro.csv
│   │   ├── qwen3_32b.csv
│   │   ├── llama_3_3_70b.csv
│   │   ├── llama_4_scout_17b.csv
│   │   ├── kimi_k2.csv
│   │   ├── llama_3_8b.csv
│   │   ├── gpt_oss_120b.csv
│   │   ├── gpt_oss_20b.csv
│   │   ├── mistral_7b.csv
│   │   ├── deepseek_r1_70b.csv
│   │   └── gemma_4_e4b.csv
│   └── aggregate/
│       └── all_model_results.csv                # Aggregated Table 6 results (12 models)
│
├── scripts/
│   ├── evaluate_new_models_v2.py                # Extended model evaluations (7 models)
│   ├── evaluate_v7_models.py                    # DeepSeek R1, Gemma 4, Gemini 2.5 Pro
│   ├── generate_figures.py                      # All paper figures + statistical outputs
│   ├── bootstrap_significance.py                # Paired bootstrap testing
│   ├── wilson_ci.py                             # 95% Wilson CI computation
│   ├── compute_kappa.py                         # IAA Cohen's kappa
│   ├── error_analysis.py                        # Error taxonomy classification
│   ├── collect_sebi.py                          # SEBI document scraper
│   ├── collect_rbi.py                           # RBI document scraper
│   ├── parse_pdfs.py                            # PDF-to-text pipeline
│   └── exp[1-11]_*.py                           # Novel methodological analysis scripts
│
├── paper/
│   ├── indiafinbench_paper_v12.md               # Current paper (v12, EMNLP 2026 submission)
│   ├── references.bib                           # BibTeX references
│   ├── figures/                                 # Publication figures (PNG + PDF)
│   └── tables/                                  # LaTeX table sources
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_kappa_analysis.ipynb
│   └── 03_evaluation_analysis.ipynb
│
├── demo/                                        # Leaderboard web application (Flask)
│   └── app.py
│
├── _archive/                                    # Intermediate/deprecated scripts (see _archive/README.md)
│
├── README.md
├── README_HF.md                                 # HuggingFace dataset card
├── requirements.txt
└── LICENSE
```

---

## Reproducing the Results

### Full Evaluation Pipeline

```bash
# 1. Collect source documents (optional — raw PDFs available on request)
python scripts/collect_sebi.py --output data/raw/sebi/
python scripts/collect_rbi.py --output data/raw/rbi/
python scripts/parse_pdfs.py --input data/raw/ --output data/parsed/

# 2. Run evaluation on a model (canonical entry point)
python evaluation/evaluate.py \
    --dataset data/benchmark/indiafinbench_v1.csv \
    --model <model_name> \
    --provider <groq|google|anthropic|ollama|openrouter> \
    --output results/predictions/<model>.csv

# 3. Compute statistics and generate figures (requires result CSVs)
python scripts/generate_figures.py

# 4. Compute inter-annotator agreement
python scripts/compute_kappa.py

# 5. Score human evaluation
python scripts/score_human_eval.py
```

All evaluation CSVs from our runs are included in `results/predictions/` and `evaluation/results/`, so steps 2 and 3 can be run independently.

### Scoring Details

Answers are scored using a four-stage procedure:
1. Exact match (after case-normalisation and punctuation stripping)
2. Fuzzy token match using RapidFuzz `token_set_ratio >= 0.72`
3. Numerical extraction match (handles currency symbols, commas, units)
4. Yes/No match for contradiction detection

The 0.72 fuzzy threshold was calibrated by manual inspection of borderline cases and validated against adjacent thresholds (0.65 too permissive, 0.80 too strict). Full ablation data in `evaluation/error_analysis/fuzzy_ablation_*.csv`.

---

## Annotation Quality

IndiaFinBench was validated through two complementary passes:

**1. Model-based secondary validation (150 items):** LLaMA-3.3-70B-Versatile independently attempted each item to verify unambiguous answerability from context. Overall agreement: 90.7%. Cohen's κ = 0.918 for contradiction detection (binary Yes/No labels).

**2. Human inter-annotator agreement (120 items across two annotator sets):** A second human annotator independently answered two stratified random samples (60 + 60 items) across all four task types, without access to the primary annotator's reference answers. Results below are from the initial 60-item set after corrected NUM scoring.

| Task | Items | Agreement | Cohen's κ |
|------|-------|-----------|-----------|
| Regulatory Interpretation | 11 | 100.0% | — |
| Temporal Reasoning | 16 | 87.5% | — |
| Contradiction Detection | 17 | 82.4% | **0.611** |
| Numerical Reasoning | 16 | 75.0% | — |
| **Overall** | **60** | **85.0%** | — |

The κ = 0.611 for contradiction detection falls in the "substantial agreement" range (Landis & Koch, 1977), consistent with human IAA on similar regulatory contradiction tasks. Full IAA data in `annotation/iaa/`.

---

## Key Findings

- **Three performance tiers**: Frontier API models (81–90%), mid-tier open-weight models (75–79%), and Gemma 4 E4B (70%). Bootstrap testing confirms most cross-tier differences are statistically significant (p<0.05).
- **Efficiency over scale**: Llama 4 Scout 17B statistically matches LLaMA-3.3-70B (p=0.79) with one-quarter the parameters.
- **Scaling plateau**: GPT-OSS 120B and GPT-OSS 20B are statistically indistinguishable (p=0.91, delta=+0.3pp).
- **DeepSeek R1 paradox**: Despite being a reasoning-specialised model, DeepSeek R1 70B ranks 11th, particularly weak on temporal reasoning (70.5%).
- **Gemini 2.5 Pro NUM weakness**: Gemini 2.5 Pro scores only 48.9% on NUM — the lowest of any model — despite ranking 1st on REG (89.7%), exposing a task-type dissociation.
- **Numerical reasoning as discriminator**: 35.9pp spread between best (Gemini 2.5 Flash: 84.8%) and worst (Gemini 2.5 Pro: 48.9%) — the most informative task for model differentiation.
- **All 12 models beat the human baseline**: Human expert accuracy was 60.0%; all 12 models exceed this, with Gemini 2.5 Flash leading at 89.7%.

---

## Retrieval-Augmented System (Extension)

This repository also includes a hybrid retrieval-augmented generation (RAG) system for querying the regulatory corpus — the open-book counterpart to the closed-book benchmark above.

**Retrieval Pipeline:** FAISS (dense) + BM25 (sparse) → Reciprocal Rank Fusion (RRF) → LLM generation

```
Corpus → Chunking → [ FAISS  +  BM25 ] → RRF → Answer
                      dense     sparse
```

### Key Result

> Hybrid retrieval improves Recall@5 from 0.687 → 0.785 (+9.7pp) on the regulatory corpus, with BM25 achieving the best MRR (0.674), confirming that citation-heavy regulatory text strongly favours lexical matching.

It was evaluated across six ablation configurations on a 35-item adversarial evaluation set.

### Results

| Config | Recall@5 | MRR | p50 ms |
|--------|----------|-----|--------|
| Dense only (B0) | 0.688 | 0.542 | 48 |
| BM25 only (B1) | 0.764 | **0.674** | 30 |
| **Hybrid RRF (B2)** ◄ | **0.785** | 0.640 | 77 |
| Small chunks 800-char (B3) | 0.583 | 0.493 | 138 |
| Large chunks 2400-char (B4) | 0.542 | 0.410 | 71 |
| Hybrid k=10 (B5) | 0.785 | 0.640 | 78 |

1600-char chunking is the empirical optimum. Smaller chunks (800) reduce recall due to fragmentation of multi-clause regulatory provisions; larger chunks (2400) introduce noise that degrades ranking quality.

This demonstrates that hybrid retrieval is necessary for domain-specific corpora containing structured identifiers such as regulatory clauses and circular references.

For full methodology, architecture, ablation analysis, and failure analysis: **[RAG_SYSTEM.md](./RAG_SYSTEM.md)**

```bash
python -m rag.scripts.build_index          # build FAISS + BM25 index (~3 min, CPU)
python -m rag.scripts.run_evaluation       # run 6-config ablation
```

---

## Citation

```bibtex
@article{pall2026indiafinbench,
  title={{IndiaFinBench}: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text},
  author={Pall, Rajveer Singh},
  journal={Proceedings of EMNLP},
  year={2026},
  url={https://github.com/Rajveer-code/IndiaFinBench}
}
```

---

## License

- **Dataset** (`data/benchmark/`, `annotation/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution
- **Code** (`scripts/`, `evaluation/`, `demo/`): [MIT License](LICENSE)
- **Source regulatory documents**: Public domain (published by Government of India agencies for public use)

---

## Contact

Rajveer Singh Pall — rajveerpall04@gmail.com

For questions about the benchmark, evaluation methodology, or collaboration inquiries, open an issue on this repository or contact directly.
