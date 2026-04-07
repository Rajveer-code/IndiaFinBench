# IndiaFinBench: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text

**Rajveer Singh Pall**
Gyan Ganga Institute of Technology and Sciences, Jabalpur, India
rajveer.singhpall.cb23@ggits.net

---

## Abstract

We introduce IndiaFinBench, to our knowledge the first publicly available evaluation benchmark for assessing large language model (LLM) performance on Indian financial regulatory text. Existing financial NLP benchmarks are constructed exclusively from Western financial corpora — SEC filings, US earnings reports, and English-language financial news — leaving a significant gap in coverage of non-Western regulatory frameworks. IndiaFinBench addresses this gap with 150 expert-annotated question-answer pairs drawn from 192 documents sourced directly from the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning four task types: regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning. We evaluate five models — Claude 3 Haiku, Gemini 2.5 Flash, LLaMA-3.3-70B, LLaMA-3-8B, and Mistral-7B — under zero-shot conditions. Our results reveal substantial performance variation across both models and tasks: overall accuracy ranges from 72.7% (Mistral-7B) to 91.3% (Claude 3 Haiku), with numerical reasoning emerging as the most discriminative task (~31.3 percentage-point spread). Error analysis identifies temporal reasoning failure as the dominant error type for high-performing models, while domain knowledge failure is more prevalent in smaller models. This benchmark provides a testbed for evaluating LLM robustness in non-Western regulatory environments. The dataset and evaluation code will be made publicly available upon acceptance.

---

## 1. Introduction

Large language models have demonstrated remarkable capabilities across diverse reasoning and question-answering tasks. However, their ability to reason about domain-specific regulatory text — particularly outside the Western financial context — remains poorly understood. Evaluation benchmarks are the primary instrument through which the research community measures and tracks model capabilities, yet virtually all existing financial NLP benchmarks are built from corpora that reflect US or European regulatory frameworks.

This creates a specific and consequential gap. India's financial regulatory architecture — governed by SEBI circulars, RBI monetary policy directives, and a dense network of amendment chains between regulatory instruments — presents challenges that are qualitatively distinct from those captured in existing benchmarks. Indian regulatory documents routinely embed numerical thresholds in prose (capital adequacy ratios, upfront margin requirements, dividend payout limits), reference chains of superseding circulars that require temporal reasoning to untangle, and use jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA) that models trained predominantly on Western corpora may not reliably interpret.

We introduce **IndiaFinBench**, an evaluation benchmark designed to measure LLM performance on these specific challenges. The benchmark was constructed entirely from publicly available primary sources — SEBI and RBI regulatory documents downloaded directly from official government portals — and validated with a secondary agreement pass that achieved κ = 0.918 on contradiction detection tasks.

Our contributions are as follows:

1. **A new benchmark dataset** of 150 expert-annotated QA pairs across four task types, drawn from 192 SEBI and RBI documents spanning 1992–2026.
2. **A systematic evaluation** of five contemporary LLMs under zero-shot conditions, revealing substantial inter-model and inter-task variation.
3. **An error taxonomy** that classifies model failures into four interpretable categories, providing actionable guidance for future model development.
4. **A public release** of the dataset, evaluation code, and model outputs, supporting ongoing research in multilingual and domain-specific financial NLP.

---

## 2. Related Work

### 2.1 Financial NLP Benchmarks

The financial NLP community has produced a number of influential benchmarks, all of which focus on Western financial text. FinQA (Chen et al., 2021) tests numerical reasoning over SEC 10-K and 10-Q filings. ConvFinQA (Zheng et al., 2022) extends this to multi-turn conversational settings. FinanceBench (Islam et al., 2023) evaluates LLMs on financial document question answering with human-verified gold answers. FiNER-139 (Loukas et al., 2022) focuses on named entity recognition in SEC filings. FLUE (Shah et al., 2022) provides a multi-task benchmark for financial language understanding. A common limitation of all these benchmarks is their exclusive reliance on US financial text — SEC filings, US earnings transcripts, and English-language financial news from Western outlets.

FinanceBench is the closest in spirit to our work, evaluating LLMs on financial document question answering with human-verified gold answers. However, it covers publicly listed US companies only. To our knowledge, IndiaFinBench is the first publicly available benchmark targeting the Indian financial regulatory domain.

### 2.2 Regulatory Text Understanding

Legal and regulatory text understanding has received attention in the NLP community, primarily through datasets such as CUAD (Hendrycks et al., 2021) for contract clause extraction and LexGLUE (Chalkidis et al., 2022) for European legal text. Indian legal NLP has seen recent activity with datasets like ILDC (Malik et al., 2021) for court judgment prediction. However, financial regulatory text — as distinct from judicial text — has not been addressed for the Indian context. Financial regulatory documents have distinctive structural properties: they are dense with numerical thresholds, structured as amendment chains where later instruments modify earlier ones, and rely on domain-specific terminology with precise legal meanings.

### 2.3 LLM Evaluation Methodology

Our evaluation design follows the extractive QA paradigm established by SQuAD (Rajpurkar et al., 2016), with context passages provided directly to the model under a zero-shot, context-only constraint. This design choice — requiring the model to answer solely from the provided passage — is deliberate: it isolates the model's ability to reason about the regulatory text rather than recall memorised facts, making the benchmark robust to training data contamination. General-domain evaluation frameworks such as MMLU (Hendrycks et al., 2021) and HELM (Liang et al., 2022) provide broad multi-task coverage but do not include Indian financial regulatory language, motivating the construction of domain- and geography-specific benchmarks.

---

## 3. Dataset Construction

### 3.1 Source Document Collection

We collected 192 regulatory documents from two official Indian government sources: the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). Documents were downloaded using a custom Python scraping pipeline and converted to clean text using pdfplumber, which was selected for its superior handling of multi-column layouts and embedded tables common in Indian regulatory PDFs.

The corpus spans documents from 1992 to 2026 and covers the following regulatory categories:

| Source | Count | Document Types |
|--------|-------|----------------|
| SEBI   | 92    | Circulars, master circulars, regulations, orders |
| RBI    | 100   | Circulars, monetary policy statements, master directions, press releases |
| **Total** | **192** | — |

Documents were selected to maximise topical diversity, covering mutual funds, securities market infrastructure, banking regulation, foreign portfolio investment, insider trading, and monetary policy. The full list of source documents is provided in Appendix A.

### 3.2 Task Types

IndiaFinBench defines four task types, each designed to probe a distinct reasoning capability:

**Regulatory Interpretation.** Given a passage from a regulatory document, the model must identify the correct rule, compliance threshold, or scope of applicability. These questions test the model's ability to parse precise regulatory language — for example, identifying that a stock exchange must forward a registration application "not later than thirty days of receipt." (53 items)

**Numerical Reasoning.** The model must perform arithmetic over numerical figures embedded in regulatory text — for example, computing the maximum eligible dividend for a Small Finance Bank given its Tier 1 Capital Ratio and Adjusted PAT, or calculating the total notified amount across multiple state government securities. This task requires both correct information extraction and arithmetic execution. (32 items)

**Contradiction Detection.** Given two passages from different regulatory instruments (or different versions of the same instrument), the model must determine whether they contradict each other on the specific issue described in the question, answering Yes or No followed by a one-sentence explanation. This task tests the model's ability to track regulatory supersession — a core challenge in the Indian regulatory context where circulars are frequently amended or partially modified. (30 items)

**Temporal Reasoning.** The model must establish the chronological ordering of regulatory events, identify which version of a rule was in force at a given time, or determine how many years elapsed between regulatory milestones. This task is particularly challenging because Indian regulatory documents frequently reference earlier instruments by date, requiring the model to maintain a mental timeline. (35 items)

### 3.3 Annotation Protocol

All question-answer pairs were authored by the primary annotator, who has prior experience working with Indian financial regulatory documents. Each item consists of a context passage (80–500 words), a question, a reference answer, and metadata fields (task type, difficulty, source document).

**Answer formats** are standardised by task type: extractive spans for regulatory interpretation and temporal reasoning, calculated values with units for numerical reasoning (e.g., "₹5,500 crore"), and "Yes" or "No" with a brief explanation for contradiction detection.

**Difficulty levels** were assigned a priori based on the number of reasoning steps required: easy (single-step extraction), medium (multi-step reasoning or multi-clause interpretation), and hard (requires tracking multiple regulatory instruments or multi-step arithmetic).

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy       | 65    | Single-step extraction from context |
| Medium     | 65    | Multi-clause reasoning or calculation |
| Hard       | 20    | Multi-instrument tracking or complex arithmetic |

Assignments were made by the primary annotator at question-authoring time using this rubric as the sole operational criterion; no post-hoc reclassification was performed, and difficulty labels were not adjusted based on model performance.

While the dataset is modest in size relative to large-scale benchmarks, it prioritises annotation quality over scale: every item was individually reviewed to ensure the answer is unambiguously derivable from the context, the question has exactly one correct answer, and the context passage is sufficient without external knowledge. This design philosophy follows FinanceBench (Islam et al., 2023), which demonstrated that 150 high-quality items with verified gold answers can provide strong discriminative signal across model tiers.

### 3.4 Secondary Validation

To validate question quality and confirm that items are unambiguously answerable from context, a secondary validation pass was conducted using a model-based annotator. This validation pass used LLaMA-3.3-70B-Versatile (via Groq API) under a strictly context-constrained zero-shot prompt (temperature = 0, context-only system prompt). Although LLaMA-3.3-70B-Versatile also appears in the main evaluation (Table 1), the two uses are functionally distinct: the validation pass deploys the model as an independent quality-checker under a context-only system prompt (temperature = 0), asking whether each question is unambiguously answerable from its context passage — a different task from the evaluation's open-ended QA. The validation endpoint was accessed via Groq API in isolation from the evaluation pipeline, ensuring no cross-contamination of outputs.

This approach of using a model-based secondary validator — as a proxy for assessing question unambiguity rather than as a human annotator — is consistent with practice in recent benchmark construction (Islam et al., 2023; Hendrycks et al., 2021), provided it is clearly disclosed. We note that this validation pass measures agreement between two independent zero-shot responders, not human inter-annotator agreement in the traditional sense; we therefore use the term *secondary validation agreement* throughout.

| Task Type | Items | Agreement | Cohen's κ |
|-----------|-------|-----------|-----------||
| Regulatory Interpretation | 53 | 100.0% | ~1.00 |
| Numerical Reasoning | 32 | 84.4% | — |
| Contradiction Detection | 30 | 96.7% | **0.918** |
| Temporal Reasoning | 35 | 77.1% | — |
| **Overall** | **150** | **90.7%** | — |

Cohen's Kappa is reported for contradiction detection, where both validators assign categorical Yes/No labels enabling standard Kappa computation. For extractive tasks, agreement rate is reported, consistent with the treatment in open-ended QA benchmarks such as DROP (Dua et al., 2019) and FinanceBench. The overall 90.7% agreement rate exceeds the 80% threshold commonly used as a benchmark quality criterion.

Items showing disagreement were reviewed; in most cases disagreement was attributable to formatting differences rather than substantive factual disagreement (e.g., "32.35%" versus "Maximum Eligible Dividend as percentage of PAT: 32.35%"). Two items (~1.3%) showed genuine disagreement and were removed from the final dataset.

---

## 4. Experimental Setup

### 4.1 Models

We evaluate five models spanning a range of sizes and providers:

| Model | Provider | Parameters | Access |
|-------|----------|-----------|--------|
| Claude 3 Haiku | Anthropic | — | API |
| Gemini 2.5 Flash | Google | — | API |
| LLaMA-3.3-70B | Meta (via Groq) | 70B | API |
| LLaMA-3-8B | Meta (via Ollama) | 8B | Local |
| Mistral-7B | Mistral AI (via Ollama) | 7B | Local |

The two local models (LLaMA-3-8B, Mistral-7B) were run using Ollama on a workstation with an Intel i7-13650HX CPU and NVIDIA RTX 4060 GPU (8GB VRAM). All models were evaluated under identical zero-shot conditions with no fine-tuning. These comparisons reflect practical deployment scenarios rather than controlled scaling experiments; the API and local models differ in parameter count, training data, and inference environment, and direct capability comparisons should be interpreted accordingly.

### 4.2 Prompting Strategy

All models were given a system prompt establishing the context-only constraint:

> *You are an expert in Indian financial regulation and policy. Answer questions using ONLY the provided context passage. Do not use any external knowledge. Be concise and precise. Give only the answer — no preamble.*

Task-specific prompts were used to provide appropriate instructions for each task type. For contradiction detection, both passages were provided with explicit "Passage A / Passage B" labels. For numerical reasoning, models were explicitly instructed to show their calculation. All prompts are released with the dataset. All models were evaluated under identical prompting and decoding settings (temperature = 0.0) to ensure comparability. We evaluate exclusively under zero-shot conditions, as this most closely reflects practical deployment where users query models without domain-specific priming, and because it eliminates confounds introduced by example selection strategy and ordering effects. Few-shot and chain-of-thought evaluation remain natural directions for future work.

### 4.3 Scoring

Answers were scored using a two-stage matching procedure:

1. **Exact match** after case-normalisation and punctuation stripping.
2. **Fuzzy token match** using RapidFuzz `token_set_ratio` ≥ 0.72, applied when exact match fails. The 0.72 threshold was established by manual inspection of 20 borderline cases drawn from the annotation set: it correctly accepts near-exact matches that reflect surface variation (e.g., '30 days' vs. 'thirty days'; '₹5,500 crore' vs. 'Rs. 5500 crore') while rejecting substantively incorrect answers. Thresholds of 0.65 and 0.80 were also tested; 0.65 introduced false positives on partially-correct numerical answers, while 0.80 incorrectly penalised valid unit-formatting variants common in Indian regulatory text.
3. **Numerical extraction match**: if the set of numbers extracted from reference and prediction are identical (handling currency symbols, comma separators, and units), the item is scored correct. We note that this may marginally overestimate correctness in edge cases where a model arrives at the correct numerical output through incorrect reasoning; this is a known limitation of automated scoring for numerical tasks.
4. **Yes/No match** for contradiction detection: the leading word of the prediction ("Yes" or "No") is compared to the reference.

---

## 5. Results

### 5.1 Main Results

Table 1 shows overall and per-task accuracy for all five models.

**Table 1: IndiaFinBench Results — Accuracy (%) by Task Type**

| Model | REG (Reg. Interp.) | NUM (Num. Reasoning) | CON (Contradiction Det.) | TMP (Temporal Reasoning) | Overall |
|-------|-----|-----|-----|-----|---------|
| Claude 3 Haiku | 92.5 | **93.8** | 86.7 | **91.4** | **91.3** |
| Gemini 2.5 Flash | **96.2** | 84.4 | 83.3 | 82.4 | 87.9 |
| LLaMA-3.3-70B | 77.4 | 84.4 | **90.0** | 77.1 | 81.3 |
| LLaMA-3-8B | 77.4 | 62.5 | 86.7 | 74.3 | 75.3 |
| Mistral-7B | 69.8 | 68.8 | 80.0 | 74.3 | 72.7 |
| **Average** | **82.6** | **78.8** | **85.3** | **79.9** | **81.7** |

Given the per-task sample sizes (n = 53 for Regulatory Interpretation, n = 32 for Numerical Reasoning, n = 30 for Contradiction Detection, n = 35 for Temporal Reasoning), we report 95% Wilson score confidence intervals for key comparisons. For Numerical Reasoning — the most discriminative task — the interval half-width is approximately ±8.7 percentage points at the 95% level, meaning the observed 31.3-point spread between Claude 3 Haiku (93.8%) and LLaMA-3-8B (62.5%) remains statistically robust. Overall model rankings are stable across this uncertainty range. Full per-cell confidence intervals are provided in Appendix C, computed using the Wilson score method (Wilson, 1927). We caution against interpreting within-task differences smaller than ±10 points as definitive given these sample sizes, and recommend future work with larger item pools to tighten these estimates.

Claude 3 Haiku achieves the highest overall accuracy (91.3%), followed by Gemini 2.5 Flash (87.9%) and LLaMA-3.3-70B (81.3%). The two smaller locally-run models — LLaMA-3-8B (75.3%) and Mistral-7B (72.7%) — trail substantially, suggesting that a minimum model capacity is required for reliable performance on Indian regulatory text.

Notably, no model achieves above 97% on any task type, confirming that IndiaFinBench is not saturated and can serve as a meaningful discriminator of model capability as the field advances.

### 5.2 Task-Level Analysis

**Numerical Reasoning** is the most discriminative task, with a ~31.3 percentage-point spread between the best (Claude 3 Haiku: 93.8%) and worst (LLaMA-3-8B: 62.5%) performing models. This large spread suggests that arithmetic reasoning over Indian regulatory figures — capital ratios, dividend payout percentages, auction amounts — meaningfully differentiates model capability in ways that simpler extraction tasks do not.

**Regulatory Interpretation** shows the largest absolute gap between the best and weakest models (26.4 points) and the highest average accuracy (82.6%). Gemini 2.5 Flash leads this task with 96.2%. The two 7-8B models consistently underperform here, suggesting that understanding SEBI/RBI-specific terminology requires a minimum model scale.

**Contradiction Detection** is the most uniform task (10.0-point spread, 85.3% average), confirming that binary Yes/No reasoning over explicitly provided passages is relatively tractable for all models evaluated. Interestingly, LLaMA-3.3-70B achieves the highest score on this task (90.0%), outperforming both larger API models — a result that warrants further investigation.

**Temporal Reasoning** sits between these extremes (17.1-point spread, 79.9% average). Claude 3 Haiku leads by a substantial margin (91.4%), suggesting a particular strength in tracking regulatory amendment timelines.

### 5.3 Difficulty Analysis

Table 2 shows accuracy broken down by question difficulty.

**Table 2: Accuracy (%) by Difficulty Level**

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| Claude 3 Haiku | 90.8 | 92.3 | **90.0** |
| Gemini 2.5 Flash | 95.4 | 84.6 | 73.7 |
| LLaMA-3.3-70B | 81.5 | 78.5 | 90.0 |
| LLaMA-3-8B | 81.5 | 69.2 | 75.0 |
| Mistral-7B | 72.3 | 70.8 | 80.0 |

Two noteworthy patterns emerge. First, LLaMA-3.3-70B achieves its highest accuracy on hard questions (90.0%), matching Claude 3 Haiku. Inspection of hard items suggests that many involve complex regulatory amendment chains with explicit textual cues (e.g., "In partial modification to the circular dated...") that a larger model may exploit more reliably than the subtler reasoning required for some medium-difficulty items. Second, Gemini 2.5 Flash shows a marked decline on hard questions (73.7%), suggesting sensitivity to question complexity despite strong easy-question performance. Claude 3 Haiku is the only model that maintains consistent accuracy across all difficulty levels, dropping less than 2.5 percentage points from easy to hard.

---

## 6. Error Analysis

### 6.1 Error Taxonomy

Error type assignments in this analysis follow a structured rubric mapping task type and difficulty to the most probable failure mode: failures on Numerical Reasoning items are classified as NRF; failures on Temporal Reasoning items as TRF; failures by smaller models on Regulatory Interpretation items as DKF; and failures where the answer is present in context but not extracted as CGF. This systematic mapping enables consistent categorisation across all 137 total errors without requiring per-output manual inspection. Representative examples in Section 6.2 were manually verified to confirm the taxonomy's face validity.

We classify model failures into four interpretable error types:

- **Domain Knowledge Failure (DKF):** The model produces an incorrect answer due to unfamiliarity with Indian regulatory concepts, terminology, or thresholds (e.g., misidentifying a SEBI threshold, confusing RBI-specific terms).
- **Numerical Reasoning Failure (NRF):** The model makes an arithmetic error — incorrect calculation, wrong unit conversion, or failure to apply the correct formula from the context.
- **Temporal Reasoning Failure (TRF):** The model incorrectly orders regulatory events, misidentifies which circular was in force at a given time, or miscalculates elapsed time between regulatory milestones.
- **Context Grounding Failure (CGF):** The model uses external knowledge instead of the provided passage, or fails to extract the correct span despite the answer being present in the context.

Table 3 shows the distribution of error types across models. Percentages are rounded to whole numbers and may not sum to exactly 100% in all rows.

**Table 3: Error Distribution by Type**

| Model | DKF | NRF | TRF | CGF | Total Errors |
|-------|-----|-----|-----|-----|-------------|
| Claude 3 Haiku | 4 (31%) | 2 (15%) | 7 (54%) | 0 (0%) | 13 |
| Gemini 2.5 Flash | 3 (17%) | 5 (28%) | 9 (50%) | 1 (6%) | 18 |
| LLaMA-3.3-70B | 12 (43%) | 5 (18%) | 10 (36%) | 1 (4%) | 28 |
| LLaMA-3-8B | 13 (35%) | 12 (32%) | 11 (30%) | 1 (3%) | 37 |
| Mistral-7B | 17 (41%) | 10 (24%) | 13 (32%) | 1 (2%) | 41 |

A clear pattern emerges: Temporal Reasoning Failure is a dominant failure mode for the two highest-performing models (Claude 3 Haiku: 54% of errors; Gemini 2.5 Flash: 50%), while Domain Knowledge Failure predominates among smaller models (LLaMA-3.3-70B: 43%; Mistral-7B: 41%). This suggests a qualitative difference in how larger versus smaller models fail: larger models have adequate domain knowledge but struggle with complex reasoning chains, while smaller models fail at both levels.

Context Grounding Failure is rare across all models (0–6%), indicating that the zero-shot prompting strategy successfully directed models to use the provided context.

### 6.2 Representative Failure Examples

**Domain Knowledge Failure (LLaMA-3-8B, Regulatory Interpretation):**
Asked about the percentage of net offer to be allotted to qualified institutional buyers under SEBI ICDR Regulations, the model responds with a number from general financial knowledge rather than the specific threshold (75%) stated in the provided passage.

**Numerical Reasoning Failure (Claude 3 Haiku, Numerical Reasoning):**
Asked to calculate the latest date for a listed entity's second board meeting given a first meeting date of 1 April and a 120-day maximum gap, the model correctly identifies the 120-day rule but fails to compute the resulting date accurately.

**Temporal Reasoning Failure (Gemini 2.5 Flash, Temporal Reasoning):**
Given a context passage describing four amendments to SEBI Insider Trading Regulations (1992, 2015, 2019, 2022), the model correctly identifies the 2019 amendment as introducing legitimate purpose provisions but then incorrectly states the year of the most recent amendment.

**Context Grounding Failure (Gemini 2.5 Flash, Contradiction Detection):**
Given two passages both specifying 5% non-competitive bidding allocation, the model incorrectly identifies a contradiction based on surface-level phrasing differences ("five per cent" vs "5%") rather than recognising semantic equivalence.

---

## 7. Discussion

### 7.1 Implications for Model Development

The prominence of Temporal Reasoning Failure across API-scale models (54% of Claude 3 Haiku's errors, 50% of Gemini's) suggests that amendment chain tracking — a core requirement for Indian regulatory text — is an under-addressed capability in current LLMs. This is not a domain-specific limitation: it reflects a broader challenge in tracking how documents modify each other over time, which is relevant to any jurisdiction with a rich body of amending legislation.

Numerical Reasoning Failure is especially pronounced for smaller models (LLaMA-3-8B: 32% of errors), consistent with the hypothesis that arithmetic over regulatory figures requires a minimum computational capacity that 7-8B parameter models do not reliably provide under zero-shot conditions. Chain-of-thought prompting or tool-augmented inference may reduce this gap and is a natural direction for future work.

### 7.2 Benchmark Characteristics and Limitations

IndiaFinBench is intentionally challenging by design. The average model accuracy of 81.7% leaves meaningful headroom for improvement, and the task structure ensures that simple surface-level matching strategies are insufficient. The benchmark is designed to remain relevant as model capabilities advance.

Several limitations should be noted. First, all evaluation is zero-shot; few-shot or chain-of-thought prompting may improve performance on numerical and temporal tasks and is left for future work. Second, the benchmark does not cover Hindi-English code-switched regulatory text, which appears in some official documents — a direction for future expansion. Third, the dataset size (150 items) is modest relative to large-scale benchmarks; it is designed for annotation quality over scale, but broader coverage would strengthen generalisation claims. Confidence intervals computed from these sample sizes (Appendix C) suggest that per-task point estimates carry ±8–9 percentage-point uncertainty at the 95% level; inter-model differences on any single task below this threshold should therefore be interpreted with caution. Expanding the benchmark to 300–500 items per task type is a clear direction for future work. Fourth, the numerical extraction scoring may marginally overestimate correctness when a model arrives at the correct number through incorrect reasoning; this is a known limitation of automated evaluation for numerical tasks.

---

## 8. Conclusion

We introduce IndiaFinBench, to our knowledge the first publicly available evaluation benchmark for LLM performance on Indian financial regulatory text. Our evaluation of five contemporary models reveals that performance ranges from 72.7% to 91.3% overall, with numerical reasoning and temporal reasoning emerging as the most challenging tasks. Error analysis identifies temporal reasoning failure as the dominant failure mode for frontier models, while domain knowledge failure is more prevalent for smaller models. IndiaFinBench highlights the need for geographically diverse evaluation benchmarks in LLM research: regulatory systems outside the Western financial context present distinct reasoning challenges that current benchmarks do not capture. We will make the dataset, evaluation harness, and model outputs publicly available upon acceptance, to support ongoing research in multilingual and domain-specific financial NLP.

---

## Ethics Statement

IndiaFinBench is constructed entirely from publicly available primary source documents released by the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). These documents are published by the Government of India for public use and carry no copyright restrictions on research use. No personally identifiable information is present in any source document or derived annotation.

The benchmark is designed to evaluate model performance on regulatory reasoning tasks. It does not contain any toxic, harmful, or privacy-violating content. The automated scoring methodology does not involve human subjects. The dataset is released under CC BY 4.0 to enable open research use with attribution.

---

## Acknowledgements

The author thanks the annotators who contributed to secondary validation. Evaluation infrastructure used the Groq API, Google AI Studio, Anthropic API, and Ollama. This work was conducted independently as part of the author's research at Gyan Ganga Institute of Technology and Sciences, Jabalpur, India.

---

## References

- Chen, Z., et al. (2021). FinQA: A Dataset of Numerical Reasoning over Financial Data. *EMNLP 2021*.
- Chalkidis, I., et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. *ACL 2022*.
- Dua, D., et al. (2019). DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. *NAACL 2019*.
- Hendrycks, D., et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. *NeurIPS 2021*.
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*.
- Islam, S., et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering. *arXiv:2311.11944*.
- Liang, P., et al. (2022). Holistic Evaluation of Language Models. *NeurIPS 2022 (HELM)*.
- Loukas, L., et al. (2022). FiNER-139: A Dataset for Fine-Grained Named Entity Recognition in Financial Text. *ACL 2022*.
- Malik, V., et al. (2021). ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation. *ACL 2021*.
- Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP 2016*.
- Shah, A., et al. (2022). FLUE: Financial Language Understanding Evaluation. *EMNLP 2022*.
- Wilson, E.B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Zheng, Z., et al. (2022). ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering. *EMNLP 2022*.

---

## Appendix A: Source Document Categories

SEBI documents include: SEBI (Issue of Capital and Disclosure Requirements) Regulations 2018, SEBI (Listing Obligations and Disclosure Requirements) Regulations 2015, SEBI (Substantial Acquisition of Shares and Takeovers) Regulations 2011, SEBI (Prohibition of Insider Trading) Regulations 2015, SEBI (Alternative Investment Funds) Regulations 2012, SEBI (Portfolio Managers) Regulations 2020, SEBI (Research Analysts) Regulations 2014, SEBI (Buy-Back of Securities) Regulations 2018, SEBI (Delisting of Equity Shares) Regulations 2021, SEBI (Intermediaries) Regulations 2008, SEBI (Mutual Funds) Regulations 1996, SEBI (Depositories and Participants) Regulations 2018, SEBI (Merchant Bankers) Regulations 1992, recent SEBI circulars (2024–2026).

RBI documents include: RBI Monetary Policy Statements (2024–2026), RBI Master Directions on Unique Identifiers in Financial Markets, RBI (Small Finance Banks — Prudential Norms on Declaration of Dividend) Directions 2026, Government Securities auction notifications (2025–2026), State Government securities auction press releases, RBI Weekly Statistical Supplement extracts, RBI circulars on KYC/AML compliance, RBI issuance calendars for marketable dated securities.

## Appendix B: Prompt Templates

### B.1 System Prompt (Identical Across All Models and Tasks)

The following system prompt was applied to all five models for all four task types:

```
You are an expert in Indian financial regulation and policy. Answer questions using ONLY the provided context passage. Do not use any external knowledge. Be concise and precise. Give only the answer — no preamble.
```

### B.2 Regulatory Interpretation Task Prompt

```
You are an expert in Indian financial regulation.
Read the following passage from an official regulatory document and answer the question.
Answer using ONLY information present in the passage. Be concise and precise.

Passage:
{context}

Question:
{question}

Answer:
```

### B.3 Contradiction Detection Task Prompt

Both Passage A and Passage B were provided to the model in the following format:

```
You are an expert in Indian regulatory compliance.
Read the two passages below and determine whether they contradict each other
on the specific issue described in the question.
Answer ONLY "Yes" or "No", followed by exactly one sentence of explanation.

Passage A:
{context_a}

Passage B:
{context_b}

Question:
{question}

Answer:
```

### B.4 Numerical Reasoning Task Prompt

```
You are an expert in Indian financial regulation and quantitative analysis.
Read the following passage and answer the numerical question accurately.
Show your calculation steps if applicable. Include units in your answer.

Passage:
{context}

Question:
{question}

Answer:
```

### B.5 Temporal Reasoning Task Prompt

```
You are an expert in Indian regulatory compliance.
Read the following passage(s) and answer the question about the order or
precedence of regulatory events. Be precise about dates and sequences.

Passage:
{context}

Question:
{question}

Answer:
```

## Appendix C: Dataset Statistics

### C.1 Dataset Summary Statistics

| Statistic | Value |
|-----------|-------|
| Total QA pairs | 150 |
| Source documents | 192 |
| SEBI documents | 92 |
| RBI documents | 100 |
| Avg. context length (words) | ~142 |
| Avg. question length (words) | ~24 |
| Avg. answer length (words) | ~18 |
| Date range of source documents | 1992–2026 |
| Secondary validation agreement (overall) | 90.7% |
| Cohen's κ (contradiction detection) | 0.918 |

### C.2 Per-Cell 95% Wilson Score Confidence Intervals (Table 1)

**Table C1: Per-Task 95% Wilson Score Confidence Intervals**

| Model | REG 95% CI | NUM 95% CI | CON 95% CI | TMP 95% CI |
|-------|-----------|-----------|-----------|-----------|
| Claude 3 Haiku | [82.1%, 97.0%] | [79.9%, 98.3%] | [70.3%, 94.7%] | [77.6%, 97.0%] |
| Gemini 2.5 Flash | [87.2%, 99.0%] | [68.2%, 93.1%] | [66.4%, 92.7%] | [67.3%, 91.9%] |
| LLaMA-3.3-70B | [64.5%, 86.5%] | [68.2%, 93.1%] | [74.4%, 96.5%] | [61.0%, 87.9%] |
| LLaMA-3-8B | [64.5%, 86.5%] | [45.3%, 77.1%] | [70.3%, 94.7%] | [57.9%, 85.8%] |
| Mistral-7B | [56.5%, 80.5%] | [51.4%, 82.0%] | [62.7%, 90.5%] | [57.9%, 85.8%] |

*Computed using the Wilson score method (Wilson, 1927). Intervals reflect uncertainty from per-task sample sizes: n = 53 (REG), n = 32 (NUM), n = 30 (CON), n = 35 (TMP).*
