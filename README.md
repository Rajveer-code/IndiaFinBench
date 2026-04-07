# IndiaFinBench

**An evaluation benchmark for large language model performance on Indian financial and regulatory text.**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-green)]()

## Overview

IndiaFinBench is a rigorously annotated evaluation benchmark of **150 expert-annotated
question-answer pairs (initial release; target 750)** drawn from SEBI circulars, RBI policy
documents, and Indian market compliance filings.

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

**Annotation:** 150 QA pairs (v1.0), annotated by primary annotator with secondary
validation. Inter-annotator agreement: Cohen's κ = 0.918 (contradiction detection),
90.7% overall.

**Task distribution:**

| Task Type | Count (v1.0) | Target (full dataset) |
|---|---|---|
| Regulatory Interpretation | 53 | ~200 |
| Numerical Reasoning | 32 | ~175 |
| Contradiction Detection | 30 | ~175 |
| Temporal Reasoning | 35 | ~200 |
| **Total** | **150** | **~750** |

## Models Evaluated

Claude 3 Haiku · Gemini 2.5 Flash · LLaMA-3.3-70B · LLaMA-3-8B · Mistral-7B

*(Results: 72.7%–91.3% overall accuracy — see paper)*

## Key Results

| Model | Overall Accuracy |
|---|---|
| Claude 3 Haiku | 91.3% |
| Gemini 2.5 Flash | 87.9% |
| LLaMA-3.3-70B | 81.3% |
| LLaMA-3-8B | 75.3% |
| Mistral-7B | 72.7% |

**Key finding:** Numerical reasoning is the most discriminative task (~31 point spread).
Temporal reasoning failure dominates frontier model errors (54% of Claude's failures).

## Project Status

| Phase | Status |
|---|---|
| Phase 1 — Data Collection (192 docs) | ✅ Complete |
| Phase 2 — Annotation (150 QA pairs, κ=0.918) | ✅ Complete |
| Phase 3 — Model Evaluation (5 models) | ✅ Complete |
| Phase 4 — Error Analysis | ✅ Complete |
| Paper v3 — Submission Ready | ✅ Complete |
| HuggingFace Dataset Release | ⬜ Upcoming |
| arXiv Preprint | ⬜ Upcoming |
| Conference Submission | ⬜ Upcoming |

**Dataset release:** Q3 2026 · **arXiv preprint:** Q3 2026

## Repository Structure
```
IndiaFinBench/
├── data/
│   ├── metadata_sebi.csv      # 92 SEBI document registry
│   ├── metadata_rbi.csv       # 100 RBI document registry
│   └── parse_report.csv       # PDF parsing results (192/192 success)
├── annotation/
│   ├── raw_qa/                # 150-item QA dataset (JSON + CSV)
│   └── guidelines/            # Annotation protocol v1
├── evaluation/
│   ├── prompts/               # Prompt templates (4 task types)
│   ├── results/               # Per-model evaluation CSVs
│   └── error_analysis/        # Error report, taxonomy, figures
├── paper/
│   ├── indiafinbench_paper_v3_submission.md   # Submission-ready manuscript
│   ├── indiafinbench_v3.tex                   # LaTeX source
│   └── references.bib                         # BibTeX references
├── scripts/
│   ├── collect_sebi.py        # SEBI document scraper
│   ├── collect_rbi.py         # RBI document scraper
│   ├── parse_pdfs.py          # PDF → clean text converter
│   ├── run_ai_annotator.py    # LLaMA-3.3-70B secondary validator
│   └── upload_to_huggingface.py  # HuggingFace dataset uploader
└── demo/
    └── app.py                 # Gradio demo for HuggingFace Spaces
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

If you use IndiaFinBench in your research, please cite:

```bibtex
@article{pall2025indiafinbench,
  title={IndiaFinBench: An Evaluation Benchmark for Large Language Model
         Performance on Indian Financial Regulatory Text},
  author={Pall, Rajveer Singh},
  journal={arXiv preprint},
  year={2025}
}
```

(Update with arXiv ID upon submission.)

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: [MIT](LICENSE)
Source documents: Public domain (Indian government publications)