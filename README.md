# IndiaFinBench

**A rigorous evaluation benchmark for large language model performance on Indian financial and regulatory text.**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-IndiaFinBench-yellow)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-v7%20Final-red)]()

---

## Overview

IndiaFinBench is a **150-item expert-annotated question-answering benchmark** drawn from SEBI circulars and RBI policy documents (192 source documents, 2000–2026). It evaluates LLM performance across four task types that are uniquely challenging in the Indian regulatory domain, where no prior benchmark exists.

**Nine contemporary LLMs** — spanning frontier API models, a dedicated reasoning model, large open-weight models, and small local models — have been evaluated. Performance ranges from **70.7% to 91.3%**, compared to a **60.0% human expert baseline**.

---

## Task Types

| Task | Code | n | Description |
|------|------|---|-------------|
| Regulatory Interpretation | REG | 53 | Compliance deadlines, applicability scopes, threshold values |
| Numerical Reasoning | NUM | 32 | Repo rate calculations, percentage changes, financial thresholds |
| Contradiction Detection | CON | 30 | Supersession of older circulars by newer ones (binary Yes/No) |
| Temporal Reasoning | TMP | 35 | Ordering and precedence of regulatory changes across amendments |
| **Total** | | **150** | |

---

## Leaderboard

**9 models evaluated** · April 2026 · Zero-shot evaluation · Sorted by overall accuracy

| Rank | Model | Size | Type | REG | NUM | CON | TMP | Overall |
|------|-------|------|------|-----|-----|-----|-----|---------|
| 1 | **Claude 3 Haiku** | — | Frontier API | 92.5% | 93.8% | 86.7% | 91.4% | **91.3%** |
| 2 | Gemini 2.5 Flash | — | Frontier API | 96.2% | 84.4% | 83.3% | 80.0% | 87.3% |
| 3 | Llama 4 Scout 17B | 17B | Open-weight API | 79.2% | 75.0% | **100.0%** | 80.0% | 82.7% |
| 3 | Qwen3-32B | 32B | Open-weight API | 77.4% | 75.0% | 86.7% | **94.3%** | 82.7% |
| 5 | LLaMA-3.3-70B | 70B | Open-weight API | 77.4% | 84.4% | 90.0% | 77.1% | 81.3% |
| 6 | LLaMA-3-8B | 8B | Local (Ollama) | 77.4% | 62.5% | 86.7% | 74.3% | 75.3% |
| 7 | Gemma 4 E4B | 4B | Local (Ollama) | 90.6% | 65.6% | 76.7% | 57.1% | 74.7% |
| 8 | Mistral-7B | 7B | Local (Ollama) | 69.8% | 68.8% | 80.0% | 74.3% | 72.7% |
| 9 | DeepSeek R1 70B | 70B distilled | Reasoning API | 60.4% | 78.1% | 93.3% | 60.0% | 70.7% |
| — | **Human Expert** *(n=30)* | — | — | 55.6% | 44.4% | 83.3% | 66.7% | 60.0% |

**Key findings:**

- Claude 3 Haiku leads at **91.3%**, outperforming the human expert baseline by **31.3 percentage points**
- Llama 4 Scout 17B achieves a perfect **100% on Contradiction Detection** — a zero-shot perfect score
- Qwen3-32B leads Temporal Reasoning at **94.3%**, suggesting strong instruction-following on amendment-chain tasks
- **Numerical Reasoning is the most discriminating task** — 31.3pp spread between best and worst
- **Reasoning model paradox:** DeepSeek R1 70B (explicit chain-of-thought) achieves the second-highest CON score (93.3%) yet ranks last overall (70.7%), with REG (60.4%) and TMP (60.0%) below several smaller non-reasoning models. Domain grounding matters more than reasoning capacity for Indian regulatory text.
- **Gemma 4 E4B** (4B parameters, local inference) achieves 90.6% on REG — competitive with frontier models on regulatory interpretation, while struggling on temporal tasks (57.1%)
- Temporal Reasoning Failure dominates frontier model error profiles (54% of Claude 3 Haiku failures)

---

## Annotation Quality

- **Source corpus:** 192 documents from official Indian government sources (sebi.gov.in, rbi.org.in)
- **Inter-annotator agreement:** Cohen's κ = 0.918 on Contradiction Detection, 90.7% overall exact agreement
- **Secondary validation:** LLaMA-3.3-70B used as AI annotator for independent verification
- **Human expert baseline:** Primary annotator answered 30 randomly sampled items (stratified by difficulty)

---

## Repository Structure

```
IndiaFinBench/
├── data/
│   ├── metadata_sebi.csv          # 92 SEBI document registry with source URLs
│   ├── metadata_rbi.csv           # 100 RBI document registry with source URLs
│   └── parse_report.csv           # PDF parsing results (192/192 success)
├── annotation/
│   ├── raw_qa/                    # 150-item QA dataset (JSON + CSV)
│   ├── guidelines/                # Annotation protocol v1
│   └── human_eval/                # 30-item human expert evaluation
├── evaluation/
│   ├── prompts/                   # Prompt templates (4 task types × zero-shot)
│   ├── results/                   # Per-model evaluation CSVs (9 models)
│   └── error_analysis/            # Error taxonomy, heatmap, difficulty breakdown
├── paper/
│   ├── indiafinbench_paper_v7_final.docx   # Submission-ready manuscript (v7)
│   ├── indiafinbench_v3.tex                # LaTeX source
│   ├── references.bib                      # BibTeX references
│   ├── figures/                            # Radar chart, heatmap, difficulty plots
│   └── tables/                             # LaTeX table sources
├── scripts/
│   ├── collect_sebi.py            # SEBI document scraper
│   ├── collect_rbi.py             # RBI document scraper
│   ├── parse_pdfs.py              # PDF → clean text converter
│   ├── evaluate.py                # Main evaluation harness (zero-shot scoring)
│   ├── compute_kappa.py           # Inter-annotator agreement computation
│   ├── error_analysis.py          # Error taxonomy and visualisations
│   ├── bootstrap_significance.py  # Paired bootstrap significance testing
│   ├── wilson_ci.py               # 95% Wilson score confidence intervals
│   ├── produce_paper_v7.py        # Paper update script (all 12 changes, v6→v7)
│   └── upload_to_huggingface.py   # HuggingFace dataset uploader
├── demo/
│   ├── app.py                     # Gradio leaderboard (HuggingFace Spaces)
│   ├── data/                      # questions.json, baselines.json
│   ├── database/                  # SQLite leaderboard backend
│   ├── evaluation/                # Live scoring engine
│   └── requirements.txt           # Demo dependencies
└── notebooks/
    ├── 01_data_exploration.ipynb       # Corpus statistics and task distribution
    ├── 02_kappa_analysis.ipynb         # Inter-annotator agreement analysis
    └── 03_evaluation_analysis.ipynb    # Model evaluation results and plots
```

---

## Reproducing the Evaluation

### Requirements
```bash
pip install -r requirements.txt
```

### Re-running evaluations
All per-item model outputs are released in `evaluation/results/`. To rerun:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_key   # for Claude
export GOOGLE_API_KEY=your_key      # for Gemini

# Evaluate a model
python scripts/evaluate.py --model claude-3-haiku-20240307 --out evaluation/results/haiku_results.csv
```

### Reproducing data collection
```bash
pip install requests beautifulsoup4 pdfplumber pandas tqdm
python scripts/collect_sebi.py   # Downloads 92 SEBI documents
python scripts/collect_rbi.py    # Downloads 100 RBI documents
python scripts/parse_pdfs.py     # Converts PDFs to clean text
```

> Note: Raw PDFs are excluded from this repo (file size). The metadata CSVs contain the original source URLs for re-downloading.

### Running the demo locally
```bash
cd demo
pip install -r requirements.txt
python app.py
```

---

## Project Status

| Phase | Status |
|-------|--------|
| Phase 1 — Data Collection (192 docs, 2 sources) | ✅ Complete |
| Phase 2 — Annotation (150 QA pairs, κ = 0.918) | ✅ Complete |
| Phase 3 — Evaluation (9 models, zero-shot) | ✅ Complete |
| Phase 4 — Error Analysis & Statistical Testing | ✅ Complete |
| Phase 5 — Paper (v7 final, 12-change revision) | ✅ Complete |
| HuggingFace Dataset Release | ⬜ Upcoming (Q3 2026) |
| arXiv Preprint | ⬜ Upcoming (Q3 2026) |
| Conference Submission (EMNLP 2026) | ⬜ Upcoming |

---

## Model Deprecation Notes

- `claude-3-haiku-20240307` was retired by Anthropic on April 19, 2026. Evaluation was completed before retirement; per-item outputs are released with the dataset for full reproducibility.
- `deepseek-r1-distill-llama-70b` was retired from Groq on October 2, 2025. Evaluated via OpenRouter, which hosts identical model weights.

---

## Citation

```bibtex
@article{pall2026indiafinbench,
  title     = {IndiaFinBench: An Evaluation Benchmark for Large Language Model
               Performance on Indian Financial Regulatory Text},
  author    = {Pall, Rajveer Singh},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

*(Update with arXiv ID upon submission.)*

---

## License

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution
- **Code:** [MIT](LICENSE)
- **Source documents:** Public domain (Indian government publications — SEBI and RBI)
