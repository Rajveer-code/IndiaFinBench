---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - finance
  - regulatory
  - india
  - sebi
  - rbi
  - benchmark
  - evaluation
pretty_name: IndiaFinBench
size_categories:
  - n<1K
---

# IndiaFinBench

**The first publicly available evaluation benchmark for LLM performance on Indian financial regulatory text.**

IndiaFinBench evaluates large language models on 406 expert-annotated question-answer pairs drawn from official SEBI and RBI regulatory documents, spanning four task types designed to probe distinct reasoning capabilities.

---

## Dataset Summary

IndiaFinBench is a zero-shot evaluation benchmark for assessing how well large language models understand Indian financial regulatory text. It contains 406 items sourced from 192 documents published by the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning the period 1992–2026. The benchmark covers four task types — regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning — calibrated to capture the specific reasoning challenges posed by Indian financial regulatory language. All items were expert-annotated with a secondary validation pass achieving κ = 0.918 on contradiction detection tasks. IndiaFinBench is intended for use as an evaluation resource; there is no train/test split — the full 406 items form the evaluation set.

---

## Supported Tasks and Leaderboard

### Task Types

| Task | Code | Items | Description |
|------|------|-------|-------------|
| **Regulatory Interpretation** | REG | 174 | Identify the correct rule, threshold, or scope of applicability from a regulatory passage |
| **Numerical Reasoning** | NUM | 92 | Perform arithmetic over numerical figures embedded in regulatory text (capital ratios, margins, dividend limits) |
| **Contradiction Detection** | CON | 62 | Determine whether two passages from different regulatory instruments contradict each other; answer Yes/No with explanation |
| **Temporal Reasoning** | TMP | 78 | Establish the ordering of regulatory events, identify which rule version was in force at a given time, or calculate elapsed time between milestones |

### Benchmark Leaderboard

Results from the full 406-item evaluation (zero-shot, context-only prompting):

| Model | REG | NUM | CON | TMP | Overall | 95% CI |
|-------|-----|-----|-----|-----|---------|--------|
| Gemini 2.5 Flash | 93.1% | 84.8% | 88.7% | 88.5% | **89.7%** | [86.3%, 92.3%] |
| Qwen3-32B | 85.1% | 77.2% | 90.3% | 92.3% | 85.5% | [81.7%, 88.6%] |
| LLaMA-3.3-70B | 86.2% | 75.0% | 95.2% | 79.5% | 83.7% | [79.8%, 87.0%] |
| Llama 4 Scout 17B | 86.2% | 66.3% | 98.4% | 84.6% | 83.3% | [79.3%, 86.6%] |
| Kimi K2 | 89.1% | 65.2% | 91.9% | 75.6% | 81.5% | [77.5%, 85.0%] |
| LLaMA-3-8B | 79.9% | 64.1% | 93.5% | 78.2% | 78.1% | [73.8%, 81.8%] |
| GPT-OSS 120B | 79.9% | 59.8% | 95.2% | 76.9% | 77.1% | [72.8%, 80.9%] |
| GPT-OSS 20B | 79.9% | 58.7% | 95.2% | 76.9% | 76.8% | [72.5%, 80.7%] |
| Mistral-7B | 79.9% | 66.3% | 80.6% | 74.4% | 75.9% | [71.5%, 79.8%] |
| DeepSeek R1 70B | 72.4% | 69.6% | 96.8% | 70.5% | 75.1% | [70.7%, 79.1%] |
| Gemma 4 E4B | 83.9% | 50.0% | 72.6% | 62.8% | 70.4% | [65.8%, 74.7%] |
| Human Expert (n=30) | 55.6% | 44.4% | 83.3% | 66.7% | 60.0% | — |

| †Claude 3 Haiku | 92.5% | 93.8% | 86.7% | 91.4% | 91.3% | [85.7%, 94.9%] |

*†Claude 3 Haiku was evaluated on the initial 150-item subset (REG=53, NUM=32, CON=30, TMP=35) due to API access constraints; not directly comparable to 406-item results.*

Confidence intervals are 95% Wilson score intervals. All model predictions and evaluation scripts are available in the [GitHub repository](https://github.com/Rajveer-code/IndiaFinBench).

---

## Languages

**English** — specifically the formal Indian legal/financial register used in SEBI and RBI regulatory documents. This register includes jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA), numerical thresholds expressed both as figures and words (e.g., "Rs. 5,500 crore", "seventy-five per cent"), and complex amendment reference chains.

---

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique item identifier (e.g., `REG_001`, `NUM_042`) |
| `task_type` | string | One of: `regulatory_interpretation`, `numerical_reasoning`, `contradiction_detection`, `temporal_reasoning` |
| `difficulty` | string | One of: `easy`, `medium`, `hard` |
| `source` | string | Source organisation: `SEBI` or `RBI` |
| `context` | string | Regulatory passage (80-500 words) from which the answer must be derived |
| `question` | string | Natural language question (~24 words on average) |
| `reference_answer` | string | Gold-standard answer (~18 words on average) |
| `source_document` | string | Document identifier linking to the source PDF |

### Data Splits

| Split | Size |
|-------|------|
| `train` (evaluation set) | 406 |

IndiaFinBench has a single split containing all 406 items. It is intended as an evaluation-only benchmark; there is no training set.

### Item Distribution

| Task Type | Items | Easy | Medium | Hard |
|-----------|-------|------|--------|------|
| Regulatory Interpretation | 174 | 62 | 85 | 27 |
| Numerical Reasoning | 92 | 36 | 40 | 16 |
| Contradiction Detection | 62 | 25 | 28 | 9 |
| Temporal Reasoning | 78 | 37 | 29 | 12 |
| **Total** | **406** | **160** | **182** | **64** |

### Example Items

**Regulatory Interpretation (easy)**
```json
{
  "id": "REG_001",
  "task_type": "regulatory_interpretation",
  "difficulty": "easy",
  "source": "SEBI",
  "context": "An issuer shall be eligible to make an initial public offer only if it has net tangible assets of at least three crore rupees in each of the preceding three full years... The issuer shall have a minimum average pre-tax operating profit of fifteen crore rupees, calculated on a restated and consolidated basis, during the three most profitable years out of the immediately preceding five years.",
  "question": "What is the minimum average pre-tax operating profit an issuer must have to be eligible for an IPO under SEBI ICDR Regulations?",
  "reference_answer": "At least fifteen crore rupees",
  "source_document": "SEBI_ICDR_Regulations_2018"
}
```

**Numerical Reasoning (hard)**
```json
{
  "id": "NUM_047",
  "task_type": "numerical_reasoning",
  "difficulty": "hard",
  "source": "RBI",
  "context": "A Small Finance Bank with a Tier 1 Capital Ratio of 15.5% (well above the minimum 7.5%) and an adjusted PAT of Rs. 320 crore for FY2025-26 proposes to declare a dividend. As per RBI directions, a SFB may declare dividends up to 33% of PAT if Tier 1 Capital Ratio exceeds 15%.",
  "question": "What is the maximum dividend this Small Finance Bank can declare?",
  "reference_answer": "Rs. 105.6 crore (33% of Rs. 320 crore PAT)",
  "source_document": "RBI_SFB_Dividend_Directions_2026"
}
```

**Contradiction Detection (medium)**
```json
{
  "id": "CON_021",
  "task_type": "contradiction_detection",
  "difficulty": "medium",
  "source": "SEBI",
  "context_a": "The listed entity shall submit the quarterly financial results within forty-five days from the end of each quarter, except for the last quarter.",
  "context_b": "The listed entity shall submit the annual audited financial results for the full financial year within sixty days from the end of the fourth quarter.",
  "question": "Do these two passages contradict each other regarding the financial results submission timeline?",
  "reference_answer": "No. The passages address different periods and are complementary.",
  "source_document": "SEBI_LODR_Regulations_2015"
}
```

**Temporal Reasoning (medium)**
```json
{
  "id": "TMP_033",
  "task_type": "temporal_reasoning",
  "difficulty": "medium",
  "source": "SEBI",
  "context": "SEBI (Prohibition of Insider Trading) Regulations were first notified in 1992. A comprehensive revision replaced the 1992 regulations entirely in 2015. The 2015 regulations were amended in 2019 to introduce the concept of 'legitimate purpose'. A further amendment in 2022 tightened the definition of 'connected persons'.",
  "question": "Which version of the insider trading regulations was in force when the concept of legitimate purpose was introduced?",
  "reference_answer": "The 2015 regulations (as amended in 2019)",
  "source_document": "SEBI_PIT_Regulations_2015"
}
```

---

## Source Data

### Who Creates the Source Data?

The source documents are official regulatory instruments published by two Indian government bodies:

1. **Securities and Exchange Board of India (SEBI)** — India's primary capital markets regulator, established in 1992. Documents include SEBI regulations, circulars, master circulars, orders, and press releases covering IPO/FPO norms, listing obligations, insider trading, mutual funds, foreign portfolio investment, alternative investment funds, and more.

2. **Reserve Bank of India (RBI)** — India's central bank. Documents include monetary policy statements, master directions, prudential norms, government securities auction notifications, and circulars on KYC/AML compliance.

### Source Document Batches

The corpus was assembled in seven batches across different regulatory domains, covering 192 documents in total from sebi.gov.in and rbi.org.in spanning 1992–2026.

---

## Annotations

### Who Annotated the Dataset?

All 406 question-answer pairs were authored by the primary researcher (Rajveer Singh Pall), with prior experience working with Indian financial regulatory documents. Each item was individually reviewed to ensure the answer is unambiguously derivable from the context and requires no external knowledge.

### Annotation Process

1. **Item authoring**: Context passages were selected from source documents; questions were authored to test specific regulatory reasoning capabilities; reference answers were written to be extractive or calculable from the context.
2. **Secondary validation**: LLaMA-3.3-70B-Versatile (Groq API) served as an independent quality-checker, validating that items are unambiguously answerable from context alone.
3. **Quality filtering**: Items with genuine disagreement (~1.3% of the initial set) were removed.

### Inter-Annotator Agreement

| Task | Agreement | Cohen's κ |
|------|-----------|-----------|
| Regulatory Interpretation | 100.0% | ~1.00 |
| Numerical Reasoning | 84.4% | — |
| Contradiction Detection | 96.7% | **0.918** |
| Temporal Reasoning | 77.1% | — |
| Overall | 90.7% | — |

Cohen's κ is reported for contradiction detection (binary Yes/No labels). Agreement rates are reported for extractive tasks, consistent with FinanceBench and DROP benchmarks.

---

## Considerations for Use

### Social Impact

This benchmark supports research in multilingual and domain-specific financial NLP. It helps researchers understand how well LLMs handle Indian regulatory text — relevant for legal technology, compliance automation, and financial advisory applications in India.

### Biases

The benchmark reflects the linguistic register of SEBI and RBI documents. Models trained predominantly on Western financial corpora may underperform relative to their general capabilities. The benchmark does not cover Hindi-English code-switched regulatory text.

### Known Limitations

- Automated scoring may marginally overestimate correctness in edge cases (especially numerical tasks)
- Human expert baseline (60.0%) reflects non-specialist annotators under time constraints, not trained domain experts
- Does not cover Hindi-English code-switched regulatory text

---

## Citation

If you use IndiaFinBench in your research, please cite:

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

- **Dataset** (`annotation/raw_qa/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Code** (`scripts/`, `evaluation/`): [MIT License](https://github.com/Rajveer-code/IndiaFinBench/blob/main/LICENSE)
- **Source documents**: Public domain (published by Government of India agencies)
