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
dataset_info:
  features:
  - name: id
    dtype: string
  - name: task_type
    dtype: string
  - name: difficulty
    dtype: string
  - name: source
    dtype: string
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: reference_answer
    dtype: string
  - name: source_document
    dtype: string
  splits:
  - name: test
    num_examples: 120
  - name: dev
    num_examples: 30
---

# IndiaFinBench

**The first publicly available evaluation benchmark for LLM performance on
Indian financial regulatory text.**

## Dataset Description

IndiaFinBench contains 150 expert-annotated question-answer pairs drawn from
192 documents sourced from the Securities and Exchange Board of India (SEBI)
and the Reserve Bank of India (RBI), spanning documents from 1992 to 2026.

### Supported Tasks

- **Regulatory Interpretation (REG):** Extract compliance rules, deadlines, and
  applicability scopes from SEBI/RBI regulatory passages.
- **Numerical Reasoning (NUM):** Perform arithmetic over financial figures embedded
  in regulatory text (capital ratios, repo rates, dividend limits).
- **Contradiction Detection (CON):** Determine whether two regulatory passages
  contradict each other (Yes/No with brief justification).
- **Temporal Reasoning (TMP):** Establish chronological ordering of regulatory
  events and identify which version of a rule was in force at a given time.

### Languages

English (Indian regulatory English — SEBI/RBI official publications)

## Dataset Structure

### Data Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique item identifier (`indiafinbench_REG_001`, etc.) |
| `task_type` | string | One of: `regulatory_interpretation`, `numerical_reasoning`, `contradiction_detection`, `temporal_reasoning` |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `source` | string | `SEBI` or `RBI` |
| `context` | string | Regulatory passage (80–500 words) |
| `question` | string | The question to answer |
| `reference_answer` | string | Gold reference answer |
| `source_document` | string | Source document identifier |

### Data Splits

| Split | Size | Notes |
|---|---|---|
| `test` | 120 | Primary evaluation split (stratified by task_type) |
| `dev`  | 30  | Development/validation split (stratified by task_type) |

Splits are reproducible: `random_state=42`, stratified on `task_type`.

### Task Distribution

| Task | N | Description |
|---|---|---|
| Regulatory Interpretation (REG) | 53 | Rule extraction, threshold identification |
| Temporal Reasoning (TMP) | 35 | Amendment sequencing, effective dates |
| Numerical Reasoning (NUM) | 32 | Arithmetic over regulatory figures |
| Contradiction Detection (CON) | 30 | Cross-passage conflict detection |

### Difficulty Distribution

| Level | N |
|---|---|
| Easy | 50 |
| Medium | 60 |
| Hard | 40 |

## Dataset Creation

### Source Data

All source documents were retrieved from official Indian government portals:

- **SEBI:** sebi.gov.in — circulars, regulations, guidelines (142 documents)
- **RBI:** rbi.org.in — monetary policy statements, master directions (89 documents but some overlap)

All source documents are in the public domain as official government publications.

### Annotations

Annotated by the dataset author (Rajveer Singh Pall) as primary annotator.
Secondary validation was performed using LLaMA-3.3-70B-Versatile (temperature=0)
as an independent annotator, achieving:

- **Overall agreement:** 90.7%
- **Cohen's κ (Contradiction Detection only):** 0.918 — binary Yes/No labels

For extractive task types (REG, NUM, TMP), fuzzy-match overlap rate
(`rapidfuzz.fuzz.partial_ratio >= 0.72`) is used as the agreement metric,
consistent with FinanceBench and DROP benchmark reporting practice.

### Annotation Guidelines

See `annotation/guidelines/` in the associated GitHub repository.

## Evaluation Results

Zero-shot evaluation (no few-shot examples, no chain-of-thought prompting):

| Model | Overall | REG | NUM | CON | TMP |
|---|---|---|---|---|---|
| Claude 3 Haiku (`claude-3-haiku-20240307`) | **91.3%** | 92.5% | 84.4% | 96.7% | 91.4% |
| Gemini 2.5 Flash (`gemini-2.5-flash`) | 87.9% | 90.6% | 81.3% | 93.3% | 85.7% |
| LLaMA-3.3-70B (Groq API) | 81.3% | 84.9% | 75.0% | 90.0% | 74.3% |
| LLaMA-3-8B (Ollama local) | 75.3% | 77.4% | 68.8% | 83.3% | 71.4% |
| Mistral-7B (Ollama local) | 72.7% | 75.5% | 62.5% | 80.0% | 71.4% |

Scoring metric: `rapidfuzz.fuzz.partial_ratio >= 0.72` (fuzzy match).

Full results, error analysis, and LaTeX tables:
[GitHub Repository](https://github.com/Rajveer-code/IndiaFinBench)

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("Rajveer-code/IndiaFinBench")

# Access splits
test_set = ds["test"]   # 120 items
dev_set  = ds["dev"]    # 30 items

# Example item
print(test_set[0])
# {
#   "id": "indiafinbench_REG_001",
#   "task_type": "regulatory_interpretation",
#   "difficulty": "easy",
#   "source": "SEBI",
#   "context": "...",
#   "question": "...",
#   "reference_answer": "...",
#   "source_document": "..."
# }
```

## Citation

If you use IndiaFinBench in your research, please cite:

```bibtex
@inproceedings{indiafinbench2026,
  title     = {{IndiaFinBench}: A Benchmark for Evaluating {LLMs}
               on Indian Financial Regulatory Text},
  author    = {Pall, Rajveer Singh},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026},
}
```

## Licensing

- **Dataset (QA pairs, annotations):** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Source documents:** Public domain (Government of India official publications)
