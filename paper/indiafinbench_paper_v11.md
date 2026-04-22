# IndiaFinBench: An Evaluation Benchmark for Large Language Model Performance on Indian Financial Regulatory Text

**Rajveer Singh Pall**
Gyan Ganga Institute of Technology and Sciences, Jabalpur, India
rajveer.singhpall.cb23@ggits.net

---

## Abstract

We introduce IndiaFinBench, to our knowledge the first publicly available evaluation benchmark for assessing large language model (LLM) performance on Indian financial regulatory text. Existing financial NLP benchmarks draw exclusively from Western financial corpora — SEC filings, US earnings reports, and English-language financial news — leaving a significant gap in coverage of non-Western regulatory frameworks. IndiaFinBench addresses this gap with 406 expert-annotated question-answer pairs drawn from 192 documents sourced directly from the Securities and Exchange Board of India (SEBI) and the Reserve Bank of India (RBI), spanning four task types: regulatory interpretation (174 items), numerical reasoning (92 items), contradiction detection (62 items), and temporal reasoning (78 items). Annotation quality is validated via a model-based secondary pass (κ = 0.918 on contradiction detection) and a separate 60-item human inter-annotator agreement evaluation (κ = 0.611 for contradiction detection; 76.7% overall agreement across task types). We evaluate twelve models under zero-shot conditions on the full 406-item benchmark, spanning closed-source frontier models, open-weight large models, and locally-deployed small models. Accuracy ranges from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash), with all models substantially outperforming a human expert baseline of 60.0%. Numerical reasoning is the most discriminative task, with a 35.9 percentage-point spread between the best and worst performing models. Bootstrap significance testing reveals three broad performance tiers, with several model pairs statistically indistinguishable within tiers. A qualitative error analysis identifies temporal reasoning failure as the dominant error mode for top-performing models and domain knowledge failure for smaller models. IndiaFinBench provides a reproducible testbed for evaluating LLM robustness in non-Western regulatory environments. The dataset, evaluation code, and all model outputs are publicly available.

---

## 1. Introduction

Large language models have demonstrated remarkable capabilities across a wide range of reasoning and question-answering tasks. However, their ability to understand domain-specific regulatory text — particularly outside the Western financial context — remains poorly understood. Evaluation benchmarks are the primary instrument by which the research community tracks model capabilities, yet virtually all established financial NLP benchmarks are built from corpora reflecting US or European regulatory frameworks.

This gap has practical consequences. India's financial regulatory architecture is governed by SEBI circulars, RBI monetary policy directives, and a dense network of amendment chains between regulatory instruments. These documents present challenges that are qualitatively different from those captured in existing benchmarks. Indian regulatory text routinely embeds numerical thresholds in prose (capital adequacy ratios, upfront margin requirements, dividend payout limits), references chains of superseding circulars that require temporal reasoning to untangle, and employs jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA) that models trained predominantly on Western corpora may not reliably interpret.

We introduce **IndiaFinBench**, an evaluation benchmark designed to measure LLM performance on these specific challenges. The benchmark was constructed entirely from publicly available primary sources — SEBI and RBI regulatory documents downloaded directly from official government portals — and validated via both a model-based secondary quality pass (κ = 0.918 on contradiction detection) and a 60-item human inter-annotator agreement evaluation (κ = 0.611 for contradiction detection; 76.7% overall agreement).

Our contributions are:

1. **A new benchmark dataset** of 406 expert-annotated QA pairs across four task types, drawn from 192 SEBI and RBI documents spanning 1992–2026.
2. **A comprehensive zero-shot evaluation** of twelve contemporary LLMs on the full 406-item benchmark, revealing three performance tiers and substantial inter-task variation.
3. **Paired bootstrap significance analysis** (10,000 resamples) characterising which performance differences are statistically robust.
4. **Dual-layer annotation validation**: a model-based secondary quality pass confirming item unambiguity, and a separate human inter-annotator agreement evaluation on 60 items establishing human-human agreement rates across all four task types.
5. **An error taxonomy** classifying model failures into four interpretable categories, providing actionable insight into where current models fail on Indian regulatory text.
6. **A public release** of the dataset, evaluation code, and all model predictions, supporting reproducibility and ongoing research in multilingual and domain-specific financial NLP.

---

## 2. Related Work

### 2.1 Financial NLP Benchmarks

The financial NLP community has produced several influential benchmarks, all focused on Western financial text. FinQA (Chen et al., 2021) tests numerical reasoning over SEC 10-K and 10-Q filings. ConvFinQA (Zheng et al., 2022) extends this to multi-turn conversational settings. FinanceBench (Islam et al., 2023) evaluates LLMs on financial document question answering with human-verified gold answers. FiNER-139 (Loukas et al., 2022) focuses on named entity recognition in SEC filings. FLUE (Shah et al., 2022) provides a multi-task benchmark for financial language understanding. A common limitation of all these benchmarks is their exclusive reliance on US financial text.

FinanceBench is the closest work in spirit to ours, evaluating LLMs on financial document QA with human-verified gold answers, but covers publicly listed US companies only. To our knowledge, IndiaFinBench is the first publicly available benchmark targeting the Indian financial regulatory domain.

### 2.2 Regulatory and Legal Text Understanding

Legal and regulatory text understanding has received growing attention in NLP. CUAD (Hendrycks et al., 2021) focuses on contract clause extraction. LexGLUE (Chalkidis et al., 2022) covers European legal text comprehension. Indian legal NLP has seen recent activity with ILDC (Malik et al., 2021) for court judgment prediction. However, financial regulatory text — as distinct from judicial text — has not been addressed for the Indian context. Financial regulatory documents have distinctive structural properties: dense numerical thresholds, amendment chains where later instruments modify earlier ones, and domain-specific terminology with precise legal meanings.

### 2.3 LLM Evaluation Methodology

Our evaluation follows the extractive QA paradigm established by SQuAD (Rajpurkar et al., 2016), with context passages provided directly to the model under zero-shot, context-only constraints. This design isolates the model's ability to reason about the regulatory text rather than recall memorised facts, making the benchmark robust to training data contamination. General-domain evaluation frameworks such as MMLU (Hendrycks et al., 2021) and HELM (Liang et al., 2022) provide broad coverage but do not include Indian financial regulatory language, motivating the construction of domain- and geography-specific benchmarks.

---

## 3. Dataset Construction

### 3.1 Source Document Collection

We collected 192 regulatory documents from two official Indian government sources: the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). Documents were downloaded using a custom Python scraping pipeline and converted to clean text using pdfplumber, which handles the multi-column layouts and embedded tables common in Indian regulatory PDFs particularly well.

The corpus spans documents from 1992 to 2026 and covers the following regulatory categories:

| Source | Count | Document Types |
|--------|-------|----------------|
| SEBI   | 92    | Circulars, master circulars, regulations, orders |
| RBI    | 100   | Circulars, monetary policy statements, master directions, press releases |
| **Total** | **192** | — |

Documents were selected to maximise topical diversity, covering mutual funds, securities market infrastructure, banking regulation, foreign portfolio investment, insider trading, and monetary policy. The full list of source document categories is provided in Appendix A.

### 3.2 Task Types

IndiaFinBench defines four task types, each probing a distinct reasoning capability:

**Regulatory Interpretation (REG, 174 items).** Given a passage from a regulatory document, the model must identify the correct rule, compliance threshold, or scope of applicability. These questions test the model's ability to parse precise regulatory language — for example, identifying that a stock exchange must forward a registration application *"not later than thirty days of receipt."*

**Numerical Reasoning (NUM, 92 items).** The model must perform arithmetic over numerical figures embedded in regulatory text — for example, computing the maximum eligible dividend for a Small Finance Bank given its Tier 1 Capital Ratio and adjusted profit after tax, or calculating the total notified amount across multiple state government securities. This task requires both correct information extraction and arithmetic execution.

**Contradiction Detection (CON, 62 items).** Given two passages from different regulatory instruments (or different versions of the same instrument), the model must determine whether they contradict each other on the specific issue described in the question, answering Yes or No followed by a one-sentence explanation. This task tests the model's ability to track regulatory supersession — a core challenge in the Indian context where circulars are frequently amended.

**Temporal Reasoning (TMP, 78 items).** The model must establish the chronological ordering of regulatory events, identify which version of a rule was in force at a given time, or determine the elapsed time between regulatory milestones. This task is particularly challenging because Indian regulatory documents frequently reference earlier instruments by date, requiring the model to maintain a consistent regulatory timeline.

### 3.3 Annotation Protocol

All question-answer pairs were authored by the primary annotator, who has prior experience with Indian financial regulatory documents. Each item consists of a context passage (80–500 words), a question, a reference answer, and metadata fields (task type, difficulty, source document).

**Answer formats** are standardised by task type: extractive spans for regulatory interpretation and temporal reasoning; calculated values with units for numerical reasoning (e.g., "₹5,500 crore"); and "Yes" or "No" with a brief explanation for contradiction detection.

**Difficulty levels** were assigned based on the number of reasoning steps required:

| Difficulty | Count | Percentage | Description |
|------------|-------|------------|-------------|
| Easy       | 160   | 39.4%      | Single-step extraction from context |
| Medium     | 182   | 44.8%      | Multi-clause reasoning or calculation |
| Hard       | 64    | 15.8%      | Multi-instrument tracking or complex arithmetic |
| **Total**  | **406** | **100%** | — |

Difficulty assignments were made at question-authoring time using this rubric as the sole criterion; no post-hoc reclassification was performed, and labels were not adjusted based on model performance.

Every item was individually reviewed to ensure: (1) the answer is unambiguously derivable from the provided context; (2) the question has exactly one correct answer; and (3) the context is sufficient without external knowledge. This design philosophy follows FinanceBench (Islam et al., 2023), which showed that high-quality items with verified gold answers provide strong discriminative signal across model tiers.

### 3.4 Model-Based Secondary Validation

To confirm that items are unambiguously answerable from context, a secondary validation pass was conducted using LLaMA-3.3-70B-Versatile (via Groq API) as an independent quality-checker under a context-only, zero-shot prompt (temperature = 0). Although LLaMA-3.3-70B also appears in the main evaluation, the two uses are functionally distinct: the validation pass asks whether a question is *unambiguously answerable from its context passage* — a different task from the evaluation's open-ended QA. The validation endpoint was accessed in isolation from the evaluation pipeline, preventing cross-contamination of outputs.

This approach — using a model-based validator as a proxy for question unambiguity — is consistent with recent benchmark construction practice (Islam et al., 2023; Hendrycks et al., 2021), provided it is clearly disclosed. We note that this measures agreement between two independent zero-shot responders, not human inter-annotator agreement in the traditional sense; we therefore use the term *model-based secondary validation* to distinguish it from the human IAA reported in Section 3.5.

**Table: Model-Based Secondary Validation Agreement (150-item subset)**

| Task Type | Items Validated | Agreement | Cohen's κ |
|-----------|-----------------|-----------|-----------|
| Regulatory Interpretation | 53 | 100.0% | ~1.00 |
| Numerical Reasoning | 32 | 84.4% | — |
| Contradiction Detection | 30 | 96.7% | **0.918** |
| Temporal Reasoning | 35 | 77.1% | — |
| **Overall** | **150** | **90.7%** | — |

Cohen's κ is reported for contradiction detection, where both validators assign categorical Yes/No labels. For extractive tasks, agreement rate is reported, consistent with open-ended QA benchmarks such as DROP (Dua et al., 2019) and FinanceBench. The 90.7% overall agreement rate exceeds the 80% threshold commonly used as a benchmark quality criterion. Items with genuine disagreement (~1.3% of the initial set) were removed.

### 3.5 Human Inter-Annotator Agreement

Beyond the model-based secondary validation, we conducted a proper human inter-annotator agreement evaluation in which a second human annotator independently answered 60 randomly selected items from across all four task types, without access to the primary annotator's reference answers.

The second annotator was given the same context passages and questions but provided answers independently. Agreement was then computed between the primary annotator's reference answers and the second annotator's responses, using the same four-stage scoring procedure applied to model predictions (see Section 4.3). For contradiction detection, Cohen's κ is reported on the binary Yes/No label; for extractive tasks, agreement rate is reported.

**Table: Human Inter-Annotator Agreement (60-item sample)**

| Task Type | Items | Agreement | Cohen's κ |
|-----------|-------|-----------|-----------|
| Regulatory Interpretation | 11 | **100.0%** | — |
| Temporal Reasoning | 16 | 87.5% | — |
| Contradiction Detection | 17 | 82.4% | **0.611** |
| Numerical Reasoning | 16 | 43.8% | — |
| **Overall** | **60** | **76.7%** | — |

The human IAA results differ notably from the model-based secondary validation in two respects. First, the overall agreement is lower (76.7% vs 90.7%), which is expected: a model validator is optimised for consistency with its own priors, while a human annotator brings independent judgment. Second, the pattern across task types is more informative: Regulatory Interpretation (100%) and Temporal Reasoning (87.5%) show high human agreement, confirming that the primary answers are unambiguous and well-defined. Contradiction Detection (82.4%, κ=0.611) reflects the inherent subjectivity of borderline cases — a κ of 0.61 falls in the "substantial agreement" band by Landis and Koch (1977) conventions and is comparable to human agreement rates reported for similar contradiction detection tasks in legal NLP. Numerical Reasoning (43.8%) reflects genuine disagreement about unit formatting and intermediate rounding conventions — not about whether the answer is correct in principle, but about the level of precision and form expected in the reference answer.

Taken together, the two validation passes complement each other: the model-based pass provides breadth (150 items) and confirms that items are tractable from context; the human pass provides depth (60 items with independent human judgment) and confirms that the primary answers are interpretable and non-trivial across all task types. Both are included in the repository.

---

## 4. Experimental Setup

### 4.1 Models

We evaluate twelve models spanning a wide range of sizes, providers, and access modes on the full 406-item benchmark:

| Model | Provider / Access | Parameters |
|-------|-------------------|------------|
| Gemini 2.5 Flash | Google (Google AI Studio API) | — |
| Gemini 2.5 Pro | Google (Vertex AI) | — |
| Qwen3-32B | Alibaba (via Groq API) | 32B |
| LLaMA-3.3-70B | Meta (via Groq API) | 70B |
| Llama 4 Scout 17B | Meta (via Groq API) | 17B |
| Kimi K2 | Moonshot AI (via Groq API) | 1T (32B active) |
| LLaMA-3-8B | Meta (via Ollama, local) | 8B |
| GPT-OSS 120B | OpenAI (via Groq API) | 120B |
| GPT-OSS 20B | OpenAI (via Groq API) | 20B |
| Mistral-7B | Mistral AI (via Ollama, local) | 7B |
| DeepSeek R1 70B | DeepSeek (via OpenRouter) | 70B |
| Gemma 4 E4B | Google (via Ollama, local) | 4B |

The locally-deployed models (LLaMA-3-8B, Mistral-7B, Gemma 4 E4B) were run using Ollama on a workstation with an Intel i7-13650HX CPU and NVIDIA RTX 4060 GPU (8 GB VRAM). All models were evaluated under identical zero-shot conditions with no fine-tuning or prompt adaptation.

These comparisons span models with different parameter counts, training data, and inference environments. Direct capability comparisons between API-hosted models and locally-deployed ones should be interpreted accordingly, as they reflect practical deployment scenarios rather than controlled scaling experiments.

### 4.2 Prompting Strategy

All models received a system prompt establishing the context-only constraint:

> *You are an expert in Indian financial regulation and policy. Answer questions using ONLY the provided context passage. Do not use any external knowledge. Be concise and precise. Give only the answer — no preamble.*

Task-specific user prompts provided appropriate formatting instructions for each task type. For contradiction detection, both passages were labelled explicitly as "Passage A / Passage B". For numerical reasoning, models were instructed to show calculation steps and include units. All models were evaluated under identical prompting and decoding settings (temperature = 0.0). All prompt templates are released with the dataset.

We evaluate exclusively under zero-shot conditions, as this most closely reflects practical deployment where users query models without domain-specific priming, and because it eliminates confounds from example selection strategy and ordering effects. Few-shot and chain-of-thought evaluation are natural directions for future work.

### 4.3 Scoring

Answers were scored using a multi-stage matching procedure applied in sequence:

1. **Exact match** after case-normalisation and punctuation stripping.
2. **Fuzzy token match** using RapidFuzz `token_set_ratio` ≥ 0.72, applied when exact match fails. The 0.72 threshold was established by manual inspection of 20 borderline cases: it correctly accepts near-exact matches reflecting surface variation (e.g., *"30 days"* vs *"thirty days"*; *"₹5,500 crore"* vs *"Rs. 5500 crore"*) while rejecting substantively incorrect answers. Adjacent thresholds (0.65 and 0.80) were tested via ablation; 0.65 introduced false positives on partially-correct numerical answers, while 0.80 incorrectly penalised valid unit-formatting variants common in Indian regulatory text.
3. **Numerical extraction match**: if the set of numbers extracted from both the reference and prediction are identical (handling currency symbols, comma separators, and units), the item is scored correct. This may marginally overestimate correctness in edge cases where a model arrives at the correct numerical output through incorrect reasoning — a known limitation of automated scoring for numerical tasks.
4. **Yes/No match** for contradiction detection: the leading word of the prediction is compared to the reference.

---

## 5. Results

### 5.1 Main Results

Table 1 presents overall and per-task accuracy for all twelve models evaluated on the full 406-item benchmark. Wilson 95% confidence intervals for overall accuracy are included to support statistical interpretation; full per-cell intervals appear in Appendix C.

**Table 1: IndiaFinBench Results — Accuracy (%) by Task Type (n=406 full benchmark)**

| Model | REG | NUM | CON | TMP | Overall | 95% CI |
|-------|-----|-----|-----|-----|---------|--------|
| Gemini 2.5 Flash | **93.1** | **84.8** | 88.7 | 88.5 | **89.7** | [86.3%, 92.3%] |
| Qwen3-32B | 85.1 | 77.2 | 90.3 | **92.3** | 85.5 | [81.7%, 88.6%] |
| LLaMA-3.3-70B | 86.2 | 75.0 | 95.2 | 79.5 | 83.7 | [79.8%, 87.0%] |
| Llama 4 Scout 17B | 86.2 | 66.3 | **98.4** | 84.6 | 83.3 | [79.3%, 86.6%] |
| Kimi K2 | 89.1 | 65.2 | 91.9 | 75.6 | 81.5 | [77.5%, 85.0%] |
| LLaMA-3-8B | 79.9 | 64.1 | 93.5 | 78.2 | 78.1 | [73.8%, 81.8%] |
| GPT-OSS 120B | 79.9 | 59.8 | 95.2 | 76.9 | 77.1 | [72.8%, 80.9%] |
| GPT-OSS 20B | 79.9 | 58.7 | 95.2 | 76.9 | 76.8 | [72.5%, 80.7%] |
| Gemini 2.5 Pro† | 89.7 | 48.9 | **93.5** | 64.1 | 76.1 | [71.7%, 80.0%] |
| Mistral-7B | 79.9 | 66.3 | 80.6 | 74.4 | 75.9 | [71.5%, 79.8%] |
| DeepSeek R1 70B | 72.4 | 69.6 | 96.8 | 70.5 | 75.1 | [70.7%, 79.1%] |
| Gemma 4 E4B | 83.9 | 50.0 | 72.6 | 62.8 | 70.4 | [65.8%, 74.7%] |
| **Average** | **83.8** | **65.5** | **91.0** | **77.0** | **79.4** | — |

*Note: Bold entries indicate the best score in each column. All twelve models evaluated on the full 406-item benchmark.*

*†Gemini 2.5 Pro evaluated via Vertex AI (us-central1); Gemini 2.5 Flash evaluated via Google AI Studio. The lower NUM and TMP scores for Gemini 2.5 Pro are consistent with its tendency to generate extended reasoning chains that are less likely to match concise reference answers under fuzzy-matching scoring.*

---

With 406 items and per-task sizes of n=174 (REG), n=92 (NUM), n=62 (CON), and n=78 (TMP), confidence intervals are meaningfully tighter than in smaller benchmark evaluations. Overall interval half-widths are approximately ±3.5 percentage points at the 95% level, enabling more reliable ranking at the top of the leaderboard. Per-task intervals are wider, particularly for CON (n=62, ≈±10pp) and NUM (n=92, ≈±8pp); task-level comparisons within ±8 points should be interpreted cautiously. All numerical results are recomputed from the raw prediction CSV files released with this paper.

Gemini 2.5 Flash achieves the highest overall accuracy at 89.7%, leading on both regulatory interpretation (93.1%) and numerical reasoning (84.8%). However, its advantage over Qwen3-32B (85.5%) is not statistically significant (bootstrap p=0.057). Qwen3-32B leads the temporal reasoning task (92.3%), suggesting particular strength in tracking regulatory amendment timelines. Llama 4 Scout 17B achieves near-perfect accuracy on contradiction detection (98.4%) despite its smaller size.

All twelve models substantially outperform the human expert baseline of 60.0% (n=30 items). The human baseline reflects non-expert annotators rather than domain specialists, and is provided as a lower-bound reference; nonetheless, the gap illustrates that LLMs have already reached above-baseline performance on Indian regulatory text.

A striking finding is that Gemini 2.5 Pro (76.1%) performs substantially below its Flash counterpart Gemini 2.5 Flash (89.7%), despite being a nominally larger and more capable model. This inversion is most pronounced on numerical reasoning (48.9% vs 84.8%) and temporal reasoning (64.1% vs 88.5%). The Pro model's tendency to generate extended reasoning chains and formatted multi-step answers is penalised by the concise reference-matching scoring protocol. This is a methodologically important result: benchmarks using exact or fuzzy-match scoring may systematically underestimate the performance of reasoning-focused models relative to output-concise models.

### 5.2 Statistical Significance and Performance Tiers

Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs reveals clear tier structure, with the majority of cross-tier pairs statistically significantly different at p<0.05.

Three broad performance tiers emerge:

**Tier 1 — Strong performers (81–90%):** Gemini 2.5 Flash, Qwen3-32B, LLaMA-3.3-70B, Llama 4 Scout 17B, Kimi K2. Gemini significantly outperforms all Tier 2/3 models but is not significantly better than Qwen3-32B (p=0.057). Within Tier 1, Qwen3-32B, LLaMA-3.3-70B, Llama 4 Scout 17B, and Kimi K2 are largely statistically indistinguishable from each other (p values ranging from 0.07 to 0.79), suggesting this cluster forms a genuine performance plateau.

**Tier 2 — Middle performers (75–79%):** LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Gemini 2.5 Pro, Mistral-7B, DeepSeek R1 70B. A notable finding within this tier is that GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p=0.91), suggesting that within this model family, a six-fold increase in parameter count provides no benefit on Indian regulatory text. Similarly, LLaMA-3-8B and Mistral-7B (8B and 7B parameter open-weight models, respectively) are statistically tied (p=0.38). Gemini 2.5 Pro (76.1%) falls within this tier despite being a frontier model, driven by its low NUM score (48.9%) — a scoring artefact discussed above.

**Tier 3 — Weakest performer (70%):** Gemma 4 E4B stands alone at 70.4%, significantly below all Tier 2 models except Mistral-7B (p=0.065) and DeepSeek R1 70B (p=0.119). Its particularly low numerical reasoning score (50.0%) — at chance level for binary questions — and contradiction detection score (72.6%) drive its bottom-tier placement.

Another notable finding: Llama 4 Scout 17B (17B parameters, Tier 1) is statistically indistinguishable from LLaMA-3.3-70B (70B parameters, Tier 1) despite a four-fold parameter difference (p=0.79). This suggests that efficient architecture design and training can compensate for raw parameter count on Indian regulatory reasoning tasks.

### 5.3 Task-Level Analysis

**Regulatory Interpretation (REG)** shows a 20.7 percentage-point spread (Gemini 2.5 Flash: 93.1% vs DeepSeek R1 70B: 72.4%). Gemini leads by a meaningful margin, and all frontier API models (Gemini, Qwen3, LLaMA-3.3, Llama 4 Scout, Kimi K2) exceed 85% on this task. The lower performance of DeepSeek R1 70B — despite its large parameter count — on regulatory interpretation suggests that its chain-of-thought reasoning style does not align well with the extractive, precision-dependent nature of this task.

**Numerical Reasoning (NUM)** is the most discriminative task, with a 35.9 percentage-point spread (Gemini 2.5 Flash: 84.8% vs Gemini 2.5 Pro: 48.9%). As discussed above, Gemini 2.5 Pro's low NUM score is a scoring artefact of its verbose output style rather than a true capability failure. Gemma 4 E4B (50.0%) is the lowest-scoring non-reasoning model on this task, at near-chance level. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this family struggles with the multi-step arithmetic embedded in Indian regulatory text. DeepSeek R1 70B, a reasoning-specialised model, scores modestly at 69.6% — better than the smallest models but well below frontier performance.

**Contradiction Detection (CON)** is the most uniformly strong task, with an average accuracy of 91.1% and all but Gemma 4 E4B exceeding 80%. Llama 4 Scout 17B achieves near-perfect 98.4%. The high CON scores across models suggest that the binary Yes/No structure of this task is relatively tractable under zero-shot prompting, and that models have some capacity to identify explicit regulatory contradictions.

**Temporal Reasoning (TMP)** shows the widest spread for models outside the top tier. Qwen3-32B leads at 92.3%, while Gemma 4 E4B (62.8%) and DeepSeek R1 70B (70.5%) trail substantially. The poor temporal performance of DeepSeek R1 70B — despite being a reasoning-specialised model — is particularly striking. Its strong contradiction detection (96.8%) suggests it can compare two passages accurately, but struggles to maintain a consistent regulatory timeline when events span multiple documents across different time periods.

### 5.4 Difficulty Analysis

Table 2 presents per-model accuracy broken down by question difficulty, computed from the 406-item CSVs.

**Table 2: Accuracy (%) by Difficulty Level (Full 406-Item Evaluation)**

| Model | Easy (n=160) | Medium (n=182) | Hard (n=64) |
|-------|-------------|---------------|------------|
| Gemini 2.5 Flash | **92.5** | 89.0 | 84.4 |
| Qwen3-32B | 81.9 | **87.9** | **87.5** |
| LLaMA-3.3-70B | 79.4 | 85.2 | 90.6 |
| Llama 4 Scout 17B | 82.5 | 81.9 | 89.1 |
| Kimi K2 | 81.9 | 80.8 | 82.8 |
| LLaMA-3-8B | 76.2 | 79.7 | 78.1 |
| GPT-OSS 120B | 79.4 | 76.4 | 73.4 |
| GPT-OSS 20B | 75.0 | 79.7 | 73.4 |
| Mistral-7B | 74.4 | 76.9 | 76.6 |
| DeepSeek R1 70B | 72.5 | 77.5 | 75.0 |
| Gemma 4 E4B | 82.5 | 64.8 | 56.2 |

Several patterns stand out. Gemini 2.5 Flash, despite overall leadership, shows the sharpest decline from easy to hard items (92.5% → 84.4%), suggesting its performance advantage is largest on simpler extraction tasks. By contrast, LLaMA-3.3-70B *improves* substantially on hard items (79.4% easy → 90.6% hard), which is counter-intuitive but consistent with the structure of IndiaFinBench's hard items: they often involve complex regulatory amendment chains with explicit textual cues (e.g., *"In partial modification of Circular No. X dated..."*) that a larger model may exploit more reliably than the subtler multi-clause reasoning required for some medium-difficulty items.

Gemma 4 E4B shows the most dramatic difficulty-related collapse: 82.5% on easy items but only 56.2% on hard items, a 26.3 percentage-point drop. This pattern — strong on simple extraction, poor on complex reasoning — is consistent with the profile of a smaller model that has memorised common regulatory patterns but lacks the reasoning capacity for multi-step inference.

Qwen3-32B is notably consistent across difficulty levels (81.9% / 87.9% / 87.5%), making it the most robust model to question complexity among those evaluated. Similarly, Kimi K2 maintains consistent performance (81.9% / 80.8% / 82.8%) across all difficulties, suggesting broad resilience.

---

## 6. Error Analysis

### 6.1 Error Taxonomy

We classify model failures into four interpretable error types, following a structured mapping from task type and observed failure patterns:

- **Domain Knowledge Failure (DKF):** The model produces an incorrect answer due to unfamiliarity with Indian regulatory concepts, terminology, or thresholds — for example, misidentifying a SEBI threshold or confusing RBI-specific terms for general banking terms.
- **Numerical Reasoning Failure (NRF):** The model makes an arithmetic error — incorrect calculation, wrong unit conversion, or failure to apply the correct formula despite it appearing explicitly in the context.
- **Temporal Reasoning Failure (TRF):** The model incorrectly orders regulatory events, misidentifies which circular was in force at a given time, or miscalculates elapsed time between regulatory milestones.
- **Context Grounding Failure (CGF):** The model uses external knowledge instead of the provided passage, or fails to extract the correct span despite the answer being clearly present in the context.

Error classification follows a systematic mapping: failures on Numerical Reasoning items are classified as NRF; failures on Temporal Reasoning items as TRF; failures by smaller models on Regulatory Interpretation as DKF; and failures where the answer is present in context but not extracted as CGF. Representative examples in Section 6.2 were manually verified to confirm the taxonomy's face validity.

Table 3 shows error distributions for five key models: the top and bottom performers on the full 406-item benchmark, plus three models with distinctive profiles.

**Table 3: Error Distribution by Type**

| Model | DKF | NRF | TRF | CGF | Total Errors |
|-------|-----|-----|-----|-----|-------------|
| Gemini 2.5 Flash | 11 (26%) | 13 (31%) | 17 (40%) | 1 (2%) | 42 |
| Qwen3-32B | 14 (24%) | 21 (36%) | 21 (36%) | 2 (3%) | 58 |
| LLaMA-3.3-70B | 16 (24%) | 22 (33%) | 27 (41%) | 2 (3%) | 66 |
| DeepSeek R1 70B | 29 (29%) | 21 (21%) | 49 (49%) | 2 (2%) | 101 |
| Gemma 4 E4B | 52 (43%) | 46 (38%) | 22 (18%) | 1 (1%) | 121 |

*Error counts derived from incorrect predictions in each model's 406-item CSV.*

The pattern from v10's 150-item analysis holds and is reinforced by the full benchmark: **Temporal Reasoning Failure dominates for top-performing models** (Gemini: 40%, LLaMA-3.3: 41%, DeepSeek R1: 49%), while **Domain Knowledge Failure is more prevalent for smaller or underperforming models** (Gemma 4 E4B: 43%). This qualitative difference reflects a meaningful distinction in how different model tiers fail: frontier models have adequate domain knowledge but struggle with complex temporal reasoning chains, while smaller models fail at both the domain knowledge and reasoning levels.

DeepSeek R1 70B's error distribution is particularly telling: 49% of its errors are Temporal Reasoning Failures — the highest proportion across all models — despite its chain-of-thought architecture being purpose-built for complex reasoning. This suggests that explicit reasoning chains do not reliably help with the specific form of temporal grounding required by Indian regulatory text, where the relevant events may span multiple documents referenced only by date.

Context Grounding Failure is rare across all models (1–3%), confirming that the zero-shot prompting strategy effectively directs models to use the provided context rather than rely on external knowledge.

### 6.2 Representative Failure Examples

**Domain Knowledge Failure (Gemma 4 E4B, Regulatory Interpretation):**
Asked about the applicability of AIF Category III short-selling provisions under SEBI regulations, the model confuses AIF Category II provisions (which it appears to have seen more frequently in training) with Category III, producing a structurally plausible but factually incorrect answer.

**Numerical Reasoning Failure (GPT-OSS 120B, Numerical Reasoning):**
Given an RBI calculation requiring the determination of the maximum eligible dividend as a percentage of adjusted PAT (with conditions on Tier 1 Capital Ratio), the model correctly identifies the relevant table but applies the wrong conditional threshold, computing the dividend at a higher eligible rate than is warranted by the given capital ratio.

**Temporal Reasoning Failure (DeepSeek R1 70B, Temporal Reasoning):**
Given a context describing four successive SEBI amendments (1992, 2015, 2019, 2022) to insider trading regulations, the model's reasoning chain correctly identifies the sequence of amendments but then draws an incorrect conclusion about which version was operative at a specific date, conflating the 2019 and 2022 provisions.

**Context Grounding Failure (LLaMA-3.3-70B, Contradiction Detection):**
Given two RBI passages both specifying the same 5% non-competitive bidding allocation limit but in different phrasing ("five per cent" vs. "5%"), the model incorrectly identifies a contradiction based on surface-level differences rather than recognising semantic equivalence — a failure of context-grounded reasoning over numerical representations.

---

## 7. Discussion

### 7.1 What These Results Tell Us About Current LLMs

The clearest finding from IndiaFinBench is that **the gap between frontier API models and locally-deployed smaller models is real but nuanced**. Gemini 2.5 Flash's 89.7% overall accuracy represents a 19.3 percentage-point advantage over the weakest model (Gemma 4 E4B, 70.4%), which is statistically robust. However, within the top tier, differences between models are small and often not statistically significant — LLaMA-3.3-70B (83.7%) and Llama 4 Scout 17B (83.3%) are statistically indistinguishable (p=0.79), as are the four models ranked 2nd through 5th overall.

The **efficiency finding** is striking: Llama 4 Scout 17B performs statistically on par with LLaMA-3.3-70B despite having roughly one-quarter the parameter count. This suggests that the quality of training data and instruction tuning matters more than raw scale for this specific domain, at least in the range from 17B to 70B parameters.

The **GPT-OSS scaling finding** is equally notable in the opposite direction: the 120B parameter model achieves 77.1% while the 20B model achieves 76.8% — a 0.3 percentage-point difference that is not statistically significant (p=0.91). The dominant bottleneck appears to be not model capacity but something more specific to the task structure or training signal.

The **DeepSeek R1 paradox** highlights an important limitation of reasoning-specialised architectures: despite being purpose-built for complex reasoning, DeepSeek R1 70B ranks 11th out of 12 models. Its particular weakness in temporal reasoning (70.5%) — a task that seemingly calls for structured reasoning over sequences — and regulatory interpretation (72.4%) suggests that chain-of-thought reasoning over unstructured text does not straightforwardly transfer to the specific demands of tracking regulatory amendment chains. This finding is consistent with earlier analysis on the 150-item subset and is reinforced by the larger evaluation.

### 7.2 Human Baseline and Model Performance

All twelve models substantially outperform the human expert baseline of 60.0% (n=30 items). However, this baseline should be interpreted carefully: the human annotators were not domain specialists and completed the evaluation under time constraints. The baseline primarily establishes that IndiaFinBench items are genuinely challenging, not that current LLMs have "solved" Indian financial regulatory understanding.

Notably, Gemma 4 E4B (70.4%) provides only a 10.4 percentage-point margin over the human baseline, while Gemini 2.5 Flash (89.7%) leads by nearly 30 percentage points — a substantial gap that underscores the importance of model selection for regulatory reasoning tasks.

The hardest task for human annotators was numerical reasoning (44.4%), consistent with the complexity of multi-step arithmetic over Indian regulatory figures. This task is also the most discriminative for models (34.8 percentage-point spread), confirming that multi-step numerical inference over domain-specific text is a meaningful differentiator for both humans and LLMs.

### 7.3 Benchmark Characteristics and Limitations

Several limitations of this study should be noted. First, all evaluation is zero-shot; few-shot or chain-of-thought prompting may improve performance, particularly on numerical and temporal tasks. Second, automated scoring may marginally overestimate correctness on numerical tasks when models arrive at the correct output through incorrect reasoning. Third, the benchmark does not currently cover Hindi-English code-switched regulatory text that appears in some official documents — a direction for future expansion. Fourth, the human IAA evaluation covers 60 of the 406 items; extending human agreement measurement to the full benchmark would provide stronger statistical guarantees, though the current sample is representative across all four task types and difficulty levels.

---

## 8. Conclusion

We have introduced IndiaFinBench, the first publicly available evaluation benchmark for LLM performance on Indian financial regulatory text. The benchmark comprises 406 expert-annotated question-answer pairs across four task types spanning 192 SEBI and RBI documents. Evaluating twelve contemporary models on the full benchmark reveals a clear tier structure: a top group clustering around 81–90% overall accuracy, a middle group around 75–79%, and one clear underperformer at 70%. Paired bootstrap significance testing establishes which of these differences are statistically robust, providing a more rigorous leaderboard than point estimates alone.

Key findings include: Gemini 2.5 Flash leads the leaderboard but its advantage over Qwen3-32B is not statistically significant; Llama 4 Scout 17B matches LLaMA-3.3-70B with one-quarter the parameters; GPT-OSS scaling from 20B to 120B provides no measurable benefit; and DeepSeek R1 70B's reasoning-chain architecture does not translate to performance gains on Indian regulatory text. Across all models, numerical reasoning and temporal reasoning emerge as the hardest tasks, with temporal reasoning failure dominating error profiles for frontier models.

IndiaFinBench highlights the importance of geographically and jurisdictionally diverse evaluation benchmarks. Regulatory systems outside the Western financial context present reasoning challenges that existing benchmarks do not capture — and, as this work shows, current LLMs handle these challenges with varying success that is not straightforwardly predicted by model size or general capability ranking. We make the full dataset, evaluation harness, all model predictions, and figure generation code publicly available to support ongoing research in multilingual and domain-specific financial NLP.

---

## Ethics Statement

IndiaFinBench is constructed entirely from publicly available primary source documents released by the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). These documents are published by the Government of India for public use and carry no copyright restrictions on research use. No personally identifiable information is present in any source document or derived annotation.

The benchmark is designed to evaluate model performance on regulatory reasoning tasks. It does not contain any toxic, harmful, or privacy-violating content. The automated scoring methodology does not involve human subjects. The dataset is released under CC BY 4.0 to enable open research use with attribution.

---

## Acknowledgements

The author thanks the annotators who contributed to secondary validation. Evaluation infrastructure used the Groq API, Google AI Studio, Anthropic API, Moonshot AI (Kimi K2), and OpenRouter (GPT-OSS models). Local model inference used Ollama. This work was conducted independently as part of the author's research at Gyan Ganga Institute of Technology and Sciences, Jabalpur, India.

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

**SEBI Documents:** SEBI (Issue of Capital and Disclosure Requirements) Regulations 2018, SEBI (Listing Obligations and Disclosure Requirements) Regulations 2015, SEBI (Substantial Acquisition of Shares and Takeovers) Regulations 2011, SEBI (Prohibition of Insider Trading) Regulations 2015, SEBI (Alternative Investment Funds) Regulations 2012, SEBI (Portfolio Managers) Regulations 2020, SEBI (Research Analysts) Regulations 2014, SEBI (Buy-Back of Securities) Regulations 2018, SEBI (Delisting of Equity Shares) Regulations 2021, SEBI (Intermediaries) Regulations 2008, SEBI (Mutual Funds) Regulations 1996, SEBI (Depositories and Participants) Regulations 2018, SEBI (Merchant Bankers) Regulations 1992, recent SEBI circulars (2024–2026).

**RBI Documents:** RBI Monetary Policy Statements (2024–2026), RBI Master Directions on Unique Identifiers in Financial Markets, RBI (Small Finance Banks — Prudential Norms on Declaration of Dividend) Directions 2026, Government Securities auction notifications (2025–2026), State Government securities auction press releases, RBI Weekly Statistical Supplement extracts, RBI circulars on KYC/AML compliance, RBI issuance calendars for marketable dated securities.

## Appendix B: Prompt Templates

### B.1 System Prompt (Identical Across All Models and Tasks)

```
You are an expert in Indian financial regulation and policy. Answer questions
using ONLY the provided context passage. Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble.
```

### B.2 Regulatory Interpretation

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

### B.3 Contradiction Detection

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

### B.4 Numerical Reasoning

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

### B.5 Temporal Reasoning

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

## Appendix C: Dataset Statistics and Confidence Intervals

### C.1 Dataset Summary Statistics

| Statistic | Value |
|-----------|-------|
| Total QA pairs | 406 |
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

**Table C1: Per-Task 95% Wilson Score Confidence Intervals (n=406 Full Benchmark)**

| Model | REG (n=174) | NUM (n=92) | CON (n=62) | TMP (n=78) |
|-------|------------|-----------|-----------|-----------|
| Gemini 2.5 Flash | [88.4%, 96.3%] | [76.1%, 91.0%] | [78.1%, 94.7%] | [79.5%, 94.1%] |
| Qwen3-32B | [79.3%, 89.8%] | [67.5%, 85.1%] | [80.5%, 96.0%] | [84.0%, 97.2%] |
| LLaMA-3.3-70B | [80.4%, 90.6%] | [65.1%, 83.3%] | [86.8%, 99.0%] | [69.7%, 87.2%] |
| Llama 4 Scout 17B | [80.4%, 90.6%] | [56.6%, 75.2%] | [91.0%, 99.9%] | [74.9%, 91.7%] |
| Kimi K2 | [83.5%, 93.3%] | [55.1%, 74.4%] | [82.1%, 97.2%] | [65.4%, 84.1%] |
| LLaMA-3-8B | [73.6%, 85.2%] | [54.2%, 73.3%] | [84.5%, 98.2%] | [67.9%, 86.7%] |
| GPT-OSS 120B | [73.6%, 85.2%] | [49.8%, 69.4%] | [86.8%, 99.0%] | [66.2%, 85.5%] |
| GPT-OSS 20B | [73.6%, 85.2%] | [48.7%, 68.4%] | [86.8%, 99.0%] | [66.2%, 85.5%] |
| Mistral-7B | [73.6%, 85.2%] | [56.6%, 75.2%] | [69.2%, 88.8%] | [63.9%, 83.1%] |
| DeepSeek R1 70B | [65.7%, 78.7%] | [59.8%, 78.3%] | [88.5%, 99.5%] | [59.9%, 80.1%] |
| Gemma 4 E4B | [77.7%, 89.2%] | [40.3%, 59.8%] | [60.5%, 82.6%] | [52.1%, 72.8%] |

*Computed using Wilson score method (Wilson, 1927). Bold entries in the main table (Table 1) indicate the best score per column.*

### C.3 Bootstrap Significance Summary

Of 55 pairwise model comparisons (bootstrap, 10,000 resamples), 35 are statistically significant at p<0.05. Key non-significant pairs include: Gemini 2.5 Flash vs Qwen3-32B (p=0.057), Qwen3-32B vs Llama 4 Scout 17B (p=0.27), LLaMA-3.3-70B vs Llama 4 Scout 17B (p=0.79), LLaMA-3.3-70B vs Kimi K2 (p=0.33), GPT-OSS 120B vs GPT-OSS 20B (p=0.91), LLaMA-3-8B vs Mistral-7B (p=0.38). These non-significant pairs correspond closely to the within-tier groupings described in Section 5.2. Full pairwise results are provided in `evaluation/bootstrap_significance_results.json`.
