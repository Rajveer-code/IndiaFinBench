---
language:
- en
license: cc-by-4.0
task_categories:
- question-answering
task_ids:
- extractive-qa
- open-domain-qa
tags:
- finance
- regulatory
- indian-law
- benchmark
- evaluation
- SEBI
- RBI
- LLM-evaluation
pretty_name: IndiaFinBench
size_categories:
- n<1K
---

# IndiaFinBench

**The first publicly available evaluation benchmark for LLM performance on
Indian financial regulatory text.**

## Dataset Description

IndiaFinBench contains 150 expert-annotated question-answer pairs drawn from
192 documents sourced from the Securities and Exchange Board of India (SEBI)
and the Reserve Bank of India (RBI), spanning documents from 1992 to 2026.

### Supported Tasks

- **Regulatory Interpretation**: Extract compliance rules, deadlines, and
  applicability scopes from SEBI/RBI regulatory passages.
- **Numerical Reasoning**: Perform arithmetic over financial figures embedded
  in regulatory text (capital ratios, repo rates, dividend limits).
- **Contradiction Detection**: Determine whether two regulatory passages
  contradict each other (Yes/No with explanation).
- **Temporal Reasoning**: Establish chronological ordering of regulatory events
  and identify which version of a rule was in force at a given time.

### Languages

English (Indian regulatory English — SEBI/RBI documents)

## Dataset Structure

### Data Fields

- `id`: unique item identifier
- `task_type`: one of regulatory_interpretation, numerical_reasoning,
  contradiction_detection, temporal_reasoning
- `difficulty`: easy, medium, or hard
- `source`: SEBI or RBI
- `context`: regulatory passage (80–500 words)
- `question`: the question
- `reference_answer`: gold answer
- `source_document`: source document identifier

### Data Splits

| Split | Size |
|---|---|
| validation | 30 |
| test | 120 |

## Dataset Creation

### Source Data

All documents sourced directly from official Indian government portals:
sebi.gov.in and rbi.org.in. All source documents are in the public domain.

### Annotations

Annotated by the dataset author (primary annotator) with secondary validation
achieving 90.7% agreement (Cohen's κ = 0.918 on contradiction detection task)
using an independent model-based validator (LLaMA-3.3-70B-Versatile, temperature=0).

### Annotation Guidelines

See annotation/guidelines/ in the associated GitHub repository.

## Evaluation Results

| Model | Overall Accuracy |
|---|---|
| Claude 3 Haiku | 91.3% |
| Gemini 2.5 Flash | 87.9% |
| LLaMA-3.3-70B | 81.3% |
| LLaMA-3-8B | 75.3% |
| Mistral-7B | 72.7% |

Full results and error analysis: [GitHub Repository](https://github.com/Rajveer-code/IndiaFinBench)

## Citation

```bibtex
@article{pall2025indiafinbench,
  title={IndiaFinBench: An Evaluation Benchmark for Large Language Model
         Performance on Indian Financial Regulatory Text},
  author={Pall, Rajveer Singh},
  journal={arXiv preprint},
  year={2025}
}
```

## Licensing

Dataset: CC BY 4.0
Source documents: Public domain (Government of India publications)
