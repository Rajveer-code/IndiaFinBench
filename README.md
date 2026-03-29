# IndiaFinBench

**An evaluation benchmark for large language model performance on Indian financial and regulatory text.**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green)]()

## Overview

IndiaFinBench is a rigorously annotated evaluation benchmark of **750 expert-verified
question-answer pairs** drawn from SEBI circulars, RBI policy documents, and Indian
market compliance filings.

It tests LLM performance on four task types that are uniquely challenging in the
Indian regulatory domain:

- **Regulatory Interpretation** — compliance deadlines, applicability scopes
- **Numerical Reasoning** — repo rate calculations, percentage changes
- **Contradiction Detection** — supersession of older circulars by newer ones
- **Temporal Reasoning** — ordering and precedence of regulatory changes

## Why This Benchmark Exists

There is no rigorous public benchmark for LLM performance on Indian financial
regulatory language. Every existing financial NLP benchmark (FinQA, ConvFinQA,
FinBench) is built on Western financial text — SEC filings, US earnings reports,
English-language financial news.

Indian regulatory text presents distinct challenges: domain-specific terminology
(LODR, PMLA, FEMA), circular supersession chains, bilingual (Hindi-English) clauses,
and regulatory logic that requires understanding the Indian legal framework.

## Dataset Construction

**Source corpus:** 192 documents collected directly from official government sources.
- 92 SEBI documents — circulars, regulations, orders, master circulars (2000–2026)
- 100 RBI documents — circulars, press releases, notifications (2024–2026)

**Annotation:** 750 QA pairs annotated by two independent annotators.
Inter-annotator agreement target: Cohen's Kappa ≥ 0.70.

**Task distribution:**
| Task Type | Count |
|---|---|
| Regulatory Interpretation | ~200 |
| Numerical Reasoning | ~175 |
| Contradiction Detection | ~175 |
| Temporal Reasoning | ~200 |

## Models Evaluated

GPT-4o · Gemini 1.5 Pro · LLaMA-3-8B · Mistral-7B · Claude 3 Haiku

*(Results to be published — evaluation in progress)*

## Project Status

| Phase | Status |
|---|---|
| Phase 1 — Data Collection (192 docs) | ✅ Complete |
| Phase 2 — Annotation (150 QA pairs) | ✅ Complete — 90.7% agreement, κ=0.918 |
| Phase 3 — Model Evaluation | ⬜ Upcoming |
| Phase 4 — Paper + HuggingFace Release | ⬜ Upcoming |

**Dataset release:** Q3 2026 · **arXiv preprint:** Q3 2026

## Repository Structure
```
IndiaFinBench/
├── data/
│   ├── metadata_sebi.csv      # 92 SEBI document registry
│   ├── metadata_rbi.csv       # 100 RBI document registry
│   └── parse_report.csv       # PDF parsing results (192/192 success)
├── annotation/
│   └── guidelines/            # Annotation protocol v1
├── evaluation/
│   └── prompts/               # Prompt templates (4 task types)
├── scripts/
│   ├── collect_sebi.py        # SEBI document scraper
│   ├── collect_rbi.py         # RBI document scraper
│   └── parse_pdfs.py          # PDF → clean text converter
└── notebooks/                 # Exploration and analysis
```

## Reproducing Data Collection
```bash
pip install requests beautifulsoup4 pdfplumber pandas tqdm
python scripts/collect_sebi.py   # Downloads SEBI documents
python scripts/collect_rbi.py    # Downloads RBI documents
python scripts/parse_pdfs.py     # Converts PDFs to clean text
```

Note: Raw PDFs are excluded from this repo (file size). The metadata CSVs
contain the original source URLs so documents can be re-downloaded.

## Citation

*BibTeX will be added upon arXiv submission (Q3 2026).*

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Code: [MIT](LICENSE)  
Source documents: Public domain (Indian government publications)