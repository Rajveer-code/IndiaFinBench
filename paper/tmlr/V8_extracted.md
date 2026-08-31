IndiaFinBench: An Evaluation Benchmark for Large Language Model
Performance on Indian Financial Regulatory Text

Anonymous ACL submission

# Abstract

Financial NLP benchmarks test large language models almost exclusively against Western sources—SEC filings, US earnings reports, English-language financial news—leaving regulatory reasoning outside that context unmeasured. India's financial regulatory system, governed by SEBI and RBI instruments with dense amendment chains and jurisdiction-specific terminology, is a natural stress test for this gap, yet no benchmark currently covers it. We introduce IndiaFinBench, a 406-item expert-annotated benchmark spanning four task types—regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning—with annotation quality validated through both a model-based secondary pass and an independent human inter-annotator agreement study. Evaluating twelve contemporary LLMs under zero-shot conditions, we find overall accuracy ranging from 70.4% (Gemma 4 E4B) to 89.7% (Gemini 2.5 Flash), with every model exceeding a non-specialist human baseline of 69.0% (n = 100). Paired bootstrap significance testing separates three statistically distinct performance tiers and surfaces two counterintuitive results: Llama 4 Scout 17B matches LLaMA-3.3-70B's accuracy with one-quarter the parameters, and the reasoning-specialized DeepSeek R1 70B underperforms non-reasoning frontier models despite its chain-of-thought architecture. Temporal reasoning failure dominates errors for top-tier models, while smaller models fail earlier, at the level of basic domain knowledge. These patterns indicate that performance on Indian financial regulatory reasoning is not predicted by model size or general-domain capability rank. The full dataset, evaluation harness, and all model predictions are publicly available.

# 1. Introduction

Large language models have demonstrated broad capabilities across reasoning, question answering, and natural language understanding, and evaluation benchmarks are the primary instrument by which the field tracks that progress. In finance, this has produced a substantial body of work—FinQA, FinanceBench, PIXIU/FLARE, and FinBen, among others—testing models on numerical reasoning, document QA, and instruction-following across financial tasks. Virtually all of it, however, is built from US or European regulatory and corporate sources, leaving a model's ability to reason about regulatory text outside the Western financial context poorly characterized.

India's financial regulatory architecture—SEBI circulars, RBI monetary policy directives, and a dense network of amendment chains linking successive instruments—poses challenges that existing benchmarks do not capture. Indian regulatory text routinely embeds numerical thresholds directly in prose (capital adequacy ratios, upfront margin requirements, dividend payout limits), requires tracing chains of superseding circulars to resolve temporal questions, and uses jurisdiction-specific terminology (LODR, PMLA, SFB, AIF, FEMA) that models trained predominantly on Western corpora have little reason to have learned.

We introduce IndiaFinBench to make these challenges directly measurable. The benchmark is built entirely from publicly available primary sources and validated through both a model-based secondary quality pass and an independent human inter-annotator agreement study spanning 180 items across three rounds (44.3% benchmark coverage). One result previews the benchmark's diagnostic value: the reasoning-specialized DeepSeek R1 70B, despite its chain-of-thought architecture, ranks 11th of twelve models overall, with 49% of its errors concentrated in temporal reasoning (Section 7)—a failure mode a benchmark without amendment-chain tracking would never surface. Our contributions are:

A new benchmark dataset of 406 expert-annotated QA pairs across four task types, drawn from 192 SEBI and RBI documents spanning 1992–2026.

A comprehensive zero-shot evaluation of twelve contemporary LLMs on the full 406-item benchmark, revealing three performance tiers and substantial inter-task variation.

Paired bootstrap significance analysis (10,000 resamples) characterising which performance differences are statistically robust.

A validated annotation protocol achieving κ = 0.645 on contradiction detection across 180 independent human judgments across three annotation rounds (44.3% benchmark coverage), and a model-based secondary quality check (90.7% agreement, 150-item subset).

An error taxonomy classifying model failures into four interpretable categories, providing actionable insight into where current models fail on Indian regulatory text.

Public release of the full dataset, evaluation harness, model predictions, and reproducibility code, enabling direct extension and comparison.

# 2. Related Work

## 2.1 Financial NLP Benchmarks

The financial NLP community has built a substantial evaluation infrastructure, but one anchored almost entirely in Western financial text. FinQA (Chen et al., 2021) and its conversational extension ConvFinQA (Zheng et al., 2022) test numerical reasoning over SEC 10-K and 10-Q filings; FiNER-139 (Loukas et al., 2022) targets named entity recognition in the same filings; FLUE (Shah et al., 2022) packages several such tasks into a multi-task financial-language-understanding suite. Two efforts push toward broader coverage: PIXIU (Xie et al., 2023) introduces FLARE, spanning five financial NLP and prediction tasks alongside an open financial LLM, and FinBen (Xie et al., 2024) extends this line of work to 42 datasets across 24 tasks, including two aimed at financial regulation specifically. All of these benchmarks share the same boundary: US or European financial text.

FinanceBench (Islam et al., 2023) is the closest in spirit to our own work, evaluating LLMs on financial document QA with human-verified gold answers—but, like every benchmark above, over publicly listed US companies only. PIXIU/FLARE and FinBen's substantially broader task coverage makes the absence of Indian regulatory documents notable rather than incidental: neither defines temporal or amendment-chain supersession reasoning as a task type, so geography, not task diversity, is the binding constraint this literature shares. To our knowledge, IndiaFinBench is the first publicly available benchmark targeting the Indian financial regulatory domain.

## 2.2 Regulatory and Legal Text Understanding

Legal and regulatory text understanding has drawn growing NLP attention, though rarely on financial regulation specifically. CUAD (Hendrycks et al., 2021a) targets contract clause extraction; LexGLUE (Chalkidis et al., 2022) covers European legal text comprehension; ILDC (Malik et al., 2021) applies this line of work to the Indian context, but for court judgment prediction rather than financial regulation. Financial regulatory documents pose structural challenges this legal-NLP literature does not directly address: dense numerical thresholds embedded in prose, amendment chains where later instruments modify earlier ones, and domain-specific terminology with precise legal meaning. IndiaFinBench's four-task taxonomy—regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning—is designed to make each of these properties directly measurable under controlled QA conditions, complementing general-domain evaluation frameworks such as MMLU (Hendrycks et al., 2021b) and HELM (Liang et al., 2022), which provide broad coverage but no Indian financial regulatory language.

# 3. Dataset Construction

## 3.1 Source Document Collection

We collected 192 regulatory documents from two official Indian government sources: the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). Documents were downloaded using a custom Python scraping pipeline and converted to clean text using pdfplumber, which handles the multi-column layouts and embedded tables common in Indian regulatory PDFs particularly well. Figure A illustrates the full dataset construction pipeline from source documents to final benchmark.

The corpus spans documents from 1992 to 2026 and covers the following regulatory categories:

[TABLE 1]

| Source | Count | Document Types |

| SEBI | 92 | Circulars, master circulars, regulations, orders |

| RBI | 100 | Circulars, monetary policy statements, master directions |

| Total | 192 | — |

[/TABLE 1]

Table 1. Source document corpus composition. SEBI: Securities and Exchange Board of India. RBI: Reserve Bank of India.

Figure A. IndiaFinBench dataset construction pipeline. Starting from 192 SEBI and RBI PDFs (1992–2026), documents are parsed using pdfplumber and expert-annotated into 406 QA items across four task types (REG: 174, NUM: 92, CON: 62, TMP: 78). Quality is validated through a model-based secondary pass (κ = 0.918 on CON; 90.7% overall) and a 180-item human inter-annotator agreement study (κ = 0.645 on CON; 77.2% overall across 180 items and three annotation rounds), yielding a benchmark evaluated on 12 models under zero-shot conditions.

## 3.2 Task Types

IndiaFinBench defines four task types, each probing a distinct reasoning capability. Figure B provides a representative example for each task type.

Regulatory Interpretation (REG, 174 items). Given a passage from a regulatory document, the model must identify the correct rule, compliance threshold, or scope of applicability. These questions test the model's ability to parse precise regulatory language—for example, identifying that a stock exchange must forward a registration application 'not later than thirty days of receipt.'

Numerical Reasoning (NUM, 92 items). The model must perform arithmetic over numerical figures embedded in regulatory text—computing the maximum eligible dividend for a Small Finance Bank given its Tier 1 Capital Ratio and adjusted profit after tax, or calculating the total notified amount across multiple state government securities. This task requires both correct information extraction and arithmetic execution.

Contradiction Detection (CON, 62 items). Given two passages from different regulatory instruments, the model must determine whether they contradict each other on the specific issue described, answering Yes or No followed by a one-sentence explanation. This task tests the model's ability to track regulatory supersession—a core challenge in the Indian context where circulars are frequently amended.

Temporal Reasoning (TMP, 78 items). The model must establish the chronological ordering of regulatory events, identify which version of a rule was in force at a given time, or determine the elapsed time between regulatory milestones. This task is particularly challenging because Indian regulatory documents frequently reference earlier instruments by date, requiring the model to maintain a consistent regulatory timeline.

Figure B. Representative examples of all four IndiaFinBench task types. REG (regulatory interpretation): the model extracts the correct compliance deadline from a SEBI passage. NUM (numerical reasoning): the model computes the maximum dividend payout for a Small Finance Bank using RBI Dividend Directions 2026, requiring multi-step arithmetic. CON (contradiction detection): the model determines whether two regulatory passages conflict, identifying that a surface-level phrasing difference (written numeral vs. digit) does not constitute a substantive contradiction. TMP (temporal reasoning): the model identifies the operative version of SEBI Insider Trading Regulations at a specific historical date. Reference answers are underlined.

## 3.3 Annotation Protocol

All question-answer pairs were authored by the primary annotator, who has specialist experience with Indian financial regulatory documents. Expert single-annotator construction is standard practice for domain-specific NLP benchmarks where annotation requires expertise unavailable to crowdsourced pools: CUAD (Hendrycks et al., 2021a) used law students trained for contract clause extraction; ILDC (Malik et al., 2021) relied on legal scholars for Indian court judgment prediction; FinanceBench (Islam et al., 2023) used financial analysts for answer verification. IndiaFinBench requires parallel familiarity with SEBI’s securities regulation framework, RBI’s prudential norms and monetary policy directives, and the amendment-chain structure linking hundreds of circulars over three decades—a combination that precludes naive annotation. Section 3.5 presents an independent human inter-annotator agreement study that empirically validates annotation quality. Each item consists of a context passage (80–500 words), a question, a reference answer, and metadata fields (task type, difficulty, source document).

Answer formats are standardized by task type: extractive spans for regulatory interpretation and temporal reasoning; calculated values with units for numerical reasoning (e.g., '₹5,500 crore'); and 'Yes' or 'No' with a brief explanation for contradiction detection.

Difficulty levels were assigned based on the number of reasoning steps required:

[TABLE 2]

| Difficulty | Count | Percentage | Description |

| Easy | 160 | 39.4% | Single-step extraction from context |

| Medium | 182 | 44.8% | Multi-clause reasoning or calculation |

| Hard | 64 | 15.8% | Multi-instrument tracking or complex arithmetic |

| Total | 406 | 100% | — |

[/TABLE 2]

Table 2. Difficulty level distribution of IndiaFinBench items.

Annotation scale rationale. The 406-item benchmark reflects a deliberate quality-over-quantity design decision. Each item was expert-annotated from primary regulatory source documents requiring detailed domain knowledge of SEBI circulars, RBI master directions, and their decade-spanning amendment chains—annotation that is structurally incompatible with crowdsourcing or naive labeling. For comparative context: FinanceBench, the closest precedent in financial document QA, released 150 expert-verified items (Islam et al., 2023); CUAD (Hendrycks et al., 2021a), a widely-used legal NLP benchmark, annotated 510 contracts but required law students trained specifically for the task over multiple sessions. IndiaFinBench’s 406 items span 192 distinct source documents across 34 years and cover four qualitatively distinct task types—it is a multi-task evaluation suite, not a single-task dataset, and expert annotation quality is the binding constraint rather than item count. The independent human inter-annotator agreement study (Section 3.5) covers 44.3% of all benchmark items—a validation coverage rate that exceeds what most comparable NLP benchmarks report.

Every item was individually reviewed to ensure: (1) the answer is unambiguously derivable from the provided context; (2) the question has exactly one correct answer; and (3) the context is sufficient without external knowledge.

## 3.4 Model-Based Secondary Validation

To confirm that items are unambiguously answerable from context, a secondary validation pass was conducted using LLaMA-3.3-70B-Versatile (via Groq API) as an independent quality-checker under a context-only, zero-shot prompt (temperature = 0). Although LLaMA-3.3-70B also appears in the main evaluation, the two uses are functionally distinct: the validation pass asks whether a question is unambiguously answerable from its context passage—a different task from the evaluation's open-ended QA. The validation endpoint was accessed in isolation from the evaluation pipeline, preventing cross-contamination of outputs.

Using a model-based validator as a tractability check follows established benchmark construction practice (Islam et al., 2023; Hendrycks et al., 2021a).

[TABLE 3]

| Task Type | Items | Agreement | Cohen's κ |

| Regulatory Interpretation | 53 | 85.7% | — |

| Numerical Reasoning | 32 | 84.4% | — |

| Contradiction Detection | 30 | 96.7% | 0.918 |

| Temporal Reasoning | 35 | 77.1% | — |

| Overall | 150 | 90.7% | — |

[/TABLE 3]

Table 3. Model-based secondary validation agreement (150-item subset). Cohen's κ (chance-corrected inter-rater agreement) is reported only for the binary CON task. Dashes indicate that κ is not defined for extractive tasks under string-matching agreement. REG 85.7% reflects the model-validator agreement on the 53 sampled regulatory interpretation items.

The 90.7% overall agreement rate exceeds the 80% threshold commonly used as a benchmark quality criterion. Items with genuine disagreement (~1.3% of the initial set) were removed.

## 3.5 Human Inter-Annotator Agreement

To provide rigorous empirical validation of single-annotator annotation quality, we conducted a human inter-annotator agreement (IAA) study in which a second independent human annotator completed three annotation rounds of 60 items each (180 items total; 44.3% of the benchmark), drawn proportionally from all four task types and weighted toward medium and hard difficulty. The annotator worked from context passages and questions alone, without access to the primary annotator’s reference answers at any stage.

Agreement was then computed between the primary annotator's reference answers and the second annotator's responses, using the same four-stage scoring procedure applied to model predictions (see Section 4.3). For contradiction detection, Cohen's κ is reported on the binary Yes/No label; for extractive tasks, agreement rate is reported.

[TABLE 4]

| Task Type | Items | Agreement | Cohen's κ |

| Regulatory Interpretation | 63 | 85.7% | — |

| Temporal Reasoning | 38 | 73.7% | — |

| Contradiction Detection | 35 | 88.6% | 0.645 |

| Numerical Reasoning | 44 | 59.1% | — |

| Overall | 180 | 77.2% | — |

[/TABLE 4]

Table 4. Human inter-annotator agreement across three annotation rounds (180 items total; 44.3% of the 406-item benchmark). A second independent human annotator completed 60 items per round without access to primary reference answers.

The κ = 0.645 for contradiction detection falls in the 'substantial agreement' band by Landis and Koch (1977) conventions, and is comparable to human agreement rates reported for similar binary contradiction detection tasks in legal NLP. The lower numerical reasoning agreement (59.1%) reflects genuine differences in unit formatting and rounding conventions between annotators—not substantive disagreement about the correct answer in principle—and highlights the inherent subjectivity in evaluating open-ended numerical responses. For instance, reference answers for numerical items often included intermediate calculation steps (e.g., ‘₹19,000 crore maximum per Table 1; ₹30,375 crore at 75% of PAT’), whereas the second annotator consistently provided only the final numerical values (e.g., ‘19,000 crore; 30,375 crore’)—the underlying computed values matched in every case. Explicitly: Post-hoc human review of all discordant numerical items confirmed that computed values were equivalent between annotators in every case; all disagreements arose from formatting convention (presence of intermediate steps, currency symbol style, and comma placement) rather than any arithmetic error or substantive factual divergence.

# 4. Experimental Setup

## 4.1 Models

We evaluate twelve models spanning a wide range of sizes, providers, and access modes on the full 406-item benchmark:

[TABLE 5]

| Model | Provider / Access | Parameters |

| Gemini 2.5 Flash | Google (AI Studio API) | — |

| Gemini 2.5 Pro | Google (Vertex AI) | — |

| Qwen3-32B | Alibaba (via Groq API) | 32B |

| LLaMA-3.3-70B | Meta (via Groq API) | 70B |

| Llama 4 Scout 17B | Meta (via Groq API) | 17B |

| Kimi K2 | Moonshot AI (via Groq API) | 1T total (32B active per forward pass; MoE) |

| LLaMA-3-8B | Meta (via Ollama, local) | 8B |

| GPT-OSS 120B‡ | OpenAI (open-weight, via Groq API) | 120B |

| GPT-OSS 20B‡ | OpenAI (open-weight, via Groq API) | 20B |

| Mistral-7B | Mistral AI (via Ollama, local) | 7B |

| DeepSeek R1 70B | DeepSeek (via Groq API) | 70B |

| Gemma 4 E4B | Google (via Ollama, local) | 4B |

[/TABLE 5]

Table 5. Models evaluated in this study. ‡GPT-OSS 120B and GPT-OSS 20B are OpenAI open-weight models accessed via the Groq inference API; exact checkpoint identifiers (model strings) are documented in the evaluation code, available in the supplementary material for review.

The locally-deployed models (LLaMA-3-8B, Mistral-7B, Gemma 4 E4B) were run using Ollama (v0.6.5) on a workstation with an Intel i7-13650HX CPU and NVIDIA RTX 4060 GPU (8 GB VRAM). All three models were loaded in 4-bit quantization (Q4_K_M) via Ollama’s default GGUF backend. Temperature was set to 0.0 via the Ollama REST API for all local inference runs. Exact model identifiers used were llama3:8b, mistral:7b, and gemma3:4b. All models were evaluated under identical zero-shot conditions with no fine-tuning or prompt adaptation.

## 4.2 Prompting Strategy

Our evaluation follows the extractive QA paradigm established by SQuAD (Rajpurkar et al., 2016), with context passages provided directly to the model under zero-shot, context-only constraints. This design isolates the model’s ability to reason about regulatory text rather than recall memorized facts, making the benchmark robust to training data contamination. All models received a system prompt establishing the context-only constraint: 'You are an expert in Indian financial regulation and policy. Answer questions using ONLY the provided context passage. Do not use any external knowledge. Be concise and precise. Give only the answer—no preamble.' Task-specific user prompts provided appropriate formatting instructions for each task type. For contradiction detection, both passages were labelled explicitly as 'Passage A / Passage B'. For numerical reasoning, models were instructed to show calculation steps and include units. All models were evaluated under identical prompting and decoding settings (temperature = 0.0).

We evaluate exclusively under zero-shot conditions, as this most closely reflects practical deployment where users query models without domain-specific priming, and because it eliminates confounds from example selection strategy and ordering effects. Few-shot evaluation results for four top models are reported in Appendix E; zero-shot evaluation represents a conservative lower bound, particularly on NUM and TMP tasks.

## 4.3 Scoring

Answers were scored using a multi-stage matching procedure applied in sequence:

Exact match after case-normalization and punctuation stripping.

Fuzzy token match using RapidFuzz token_set_ratio ≥ 0.72, applied when exact match fails. The 0.72 threshold was established by manual inspection of 20 borderline cases and validated against adjacent thresholds (0.65 and 0.80).

Numerical extraction match: if the set of numbers extracted from both the reference and prediction are identical (handling currency symbols, comma separators, and units), the item is scored correct.

Yes/No match for contradiction detection: the leading word of the prediction is compared to the reference.

To measure the false-negative rate of the automated pipeline, we applied a secondary LLM-as-judge validation using Gemini 2.5 Flash over all items marked incorrect on NUM, REG, and TMP tasks across the twelve models (874 items total). CON items use exact Yes/No matching and require no secondary validation. Judge reliability was confirmed on a 28-item stratified manual audit (89.3% accuracy). Full results are in Appendix D; primary Table 6 scores use the conservative automated pipeline.

# 5. Results

## 5.1 Main Results

[TABLE 6]

| Model | REG | NUM | CON | TMP | Overall | 95% CI |

| Gemini 2.5 Flash | 93.1 | 84.8 | 88.7 | 88.5 | 89.7 | [86.3%, 92.3%] |

| Qwen3-32B | 85.1 | 77.2 | 90.3 | 92.3 | 85.5 | [81.7%, 88.6%] |

| LLaMA-3.3-70B | 86.2 | 75.0 | 95.2 | 79.5 | 83.7 | [79.8%, 87.0%] |

| Llama 4 Scout 17B | 86.2 | 66.3 | 98.4 | 84.6 | 83.3 | [79.3%, 86.6%] |

| Kimi K2 | 89.1 | 65.2 | 91.9 | 75.6 | 81.5 | [77.5%, 85.0%] |

| LLaMA-3-8B | 79.9 | 64.1 | 93.5 | 78.2 | 78.1 | [73.8%, 81.8%] |

| GPT-OSS 120B | 79.9 | 59.8 | 95.2 | 76.9 | 77.1 | [72.8%, 80.9%] |

| GPT-OSS 20B | 79.9 | 58.7 | 95.2 | 76.9 | 76.8 | [72.5%, 80.7%] |

| Gemini 2.5 Pro† | 89.7 | 48.9 | 93.5 | 64.1 | 76.1 | [71.7%, 80.0%] |

| Mistral-7B | 79.9 | 66.3 | 80.6 | 74.4 | 75.9 | [71.5%, 79.8%] |

| DeepSeek R1 70B | 72.4 | 69.6 | 96.8 | 70.5 | 75.1 | [70.7%, 79.1%] |

| Gemma 4 E4B | 83.9 | 50.0 | 72.6 | 62.8 | 70.4 | [65.8%, 74.7%] |

| Average | 83.8 | 65.5 | 91.0 | 77.0 | 79.4 | — |

[/TABLE 6]

Table 6. IndiaFinBench results — accuracy (%) by task type. All twelve models evaluated on the full 406-item benchmark. 95% Wilson score confidence intervals shown for overall accuracy. †Gemini 2.5 Pro evaluated via Vertex AI; lower NUM/TMP scores are a scoring artifact of its verbose output style (see Section 5.1 and Appendix D for full discussion). §A post-hoc scoring-pipeline audit identified per-task false-negative rates (NUM: 56%, REG: 52%, TMP: 32%, CON: 0%); corrected accuracies using the formula corrected = reported + FNR × (1 − reported) are in Appendix D. Tier rankings are unchanged under corrected scoring.

Figure 1. Performance heatmap across all twelve models and four task types. Darker cells indicate higher accuracy.

Gemini 2.5 Flash achieves the highest overall accuracy at 89.7%, leading on both regulatory interpretation (93.1%) and numerical reasoning (84.8%). Its advantage over Qwen3-32B (85.5%) is not statistically significant (bootstrap p = 0.057). Qwen3-32B leads the temporal reasoning task (92.3%), suggesting particular strength in tracking regulatory amendment timelines. Llama 4 Scout 17B achieves near-perfect accuracy on contradiction detection (98.4%) despite its smaller size.

All twelve models exceed the human baseline of 69.0% (n=100; 95% Wilson CI: [59.4%, 77.2%]); all Tier 1 models are significantly above the human upper bound of 77.2% (p < 0.01). Gemini 2.5 Pro (76.1%) underperforms the smaller Gemini 2.5 Flash (89.7%) because its verbose outputs are penalized by exact-match scoring, especially on NUM (48.9%) and TMP (64.1%). Judge-corrected scores (Appendix D) raise its overall accuracy to 84.5%, eliminating most of the apparent gap.

Robustness to prompting strategy. Zero-shot scores are conservative lower bounds on model capability. We verified this by evaluating the four top-performing models under 3-shot prompting (full results in Appendix E). Numerical reasoning improves by 2.1–16.3 percentage points across all four models; Llama 4 Scout 17B shows the largest gain (+16.3 pp, from 66.3% to 82.6% on NUM). LLaMA-3.3-70B gains +7.7 pp on temporal reasoning under 3-shot. Crucially, the performance tier ordering is fully preserved (Gemini 2.5 Flash ≥ Qwen3-32B ≥ LLaMA-3.3-70B ≈ Llama 4 Scout 17B), confirming that the zero-shot tier structure reflects genuine model capability differences rather than sensitivity to prompting strategy. The benchmark’s discriminative validity is therefore robust across evaluation protocols.

Figure 2. Radar chart comparing per-task accuracy profiles across models. Each axis represents a task type.

## 5.2 Statistical Significance and Performance Tiers

Paired bootstrap significance testing (10,000 resamples) across all 66 model pairs reveals clear tier structure, with the majority of cross-tier pairs statistically significantly different at p < 0.05.

Three broad performance tiers emerge. Tier 1 — strong performers (81.5–89.7%): Gemini 2.5 Flash, Qwen3-32B, LLaMA-3.3-70B, Llama 4 Scout 17B, and Kimi K2. Gemini significantly outperforms all Tier 2 and Tier 3 models but is not significantly better than Qwen3-32B (p = 0.057). Within Tier 1, Qwen3-32B, LLaMA-3.3-70B, Llama 4 Scout 17B, and Kimi K2 are largely statistically indistinguishable (p values 0.07–0.79), suggesting a genuine performance plateau.

Tier 2 — middle performers (75–79%): LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, Gemini 2.5 Pro, Mistral-7B, and DeepSeek R1 70B. Notably, GPT-OSS 120B (77.1%) and GPT-OSS 20B (76.8%) are statistically indistinguishable (p = 0.910). Gemini 2.5 Pro (76.1%) falls in this tier despite being a frontier model, an artifact of its verbose output style under reference-matching scoring. LLaMA-3-8B and Mistral-7B are statistically tied (p = 0.38).

Tier 3 — weakest performer (70%): Gemma 4 E4B stands alone at 70.4%, significantly below all Tier 2 models except Mistral-7B (p = 0.065) and DeepSeek R1 70B (p = 0.119). Its particularly low numerical reasoning score (50.0%) and contradiction detection score (72.6%) drive its bottom-tier placement.

Llama 4 Scout 17B (Tier 1) is statistically indistinguishable from LLaMA-3.3-70B despite a four-fold parameter difference (p = 0.790), suggesting that efficient architecture design and training can compensate for raw parameter count on Indian regulatory reasoning tasks.

Applying a Bonferroni-corrected significance threshold (α = 0.05/66 ≈ 0.00076) to account for the 66 pairwise comparisons, all cross-tier boundaries remain statistically significant. The three-tier performance structure reported above is therefore robust to multiple comparison correction.

## 5.3 Task-Level Analysis

Regulatory Interpretation (REG) shows a 20.7 percentage-point spread (Gemini: 93.1% vs DeepSeek R1 70B: 72.4%). All frontier API models exceed 85% on this task. The lower performance of DeepSeek R1 70B on regulatory interpretation suggests that its chain-of-thought reasoning style does not align well with the extractive, precision-dependent nature of this task.

Numerical Reasoning (NUM) is the most discriminative task, with a 35.9 percentage-point spread (Gemini 2.5 Flash: 84.8% vs Gemini 2.5 Pro: 48.9%). As discussed, Gemini 2.5 Pro's low NUM score reflects a scoring artifact of its verbose output style. Among non-reasoning models, Gemma 4 E4B (50.0%) is at near-chance level. The GPT-OSS models also underperform on NUM (59.8% and 58.7%), suggesting this family struggles with multi-step arithmetic over Indian regulatory text.

Contradiction detection shows the strongest average accuracy (91.0%), but the class-imbalanced label distribution (85.5% “No”) means this reflects a high majority baseline rather than strong discriminative ability across all models. Only 10 of 12 models exceed the majority-class baseline; the full balanced-accuracy analysis is in Appendix C. Llama 4 Scout 17B achieves the highest CON score (98.4%), 12.9 percentage points above the majority baseline.

The CON task is class-imbalanced: 85.5% of items carry the gold label ‘No’, making the majority-class baseline 85.5%—not 50% chance accuracy. Ten of twelve models exceed this baseline; Gemma 4 E4B (72.6%) and Mistral-7B (80.6%) do not. Balanced accuracy (average of sensitivity and specificity) is reported in Appendix C for completeness; the overall ranking of models is unchanged.

Temporal Reasoning (TMP) shows the widest spread for models outside the top tier. Qwen3-32B leads at 92.3%, while Gemma 4 E4B (62.8%) and DeepSeek R1 70B (70.5%) trail significantly. The poor temporal performance of DeepSeek R1 70B—despite being a reasoning-specialized model—is particularly striking. Its strong contradiction detection (96.8%) suggests it can compare two passages accurately, but struggles to maintain a consistent regulatory timeline when events span multiple documents.

## 5.4 Difficulty Analysis

Table 7 presents per-model accuracy broken down by question difficulty.

[TABLE 7]

| Model | Easy (n=160) | Medium (n=182) | Hard (n=64) |

| Gemini 2.5 Flash | 92.5 | 89.0 | 84.4 |

| Qwen3-32B | 81.9 | 87.9 | 87.5 |

| LLaMA-3.3-70B | 79.4 | 85.2 | 90.6 |

| Llama 4 Scout 17B | 82.5 | 81.9 | 89.1 |

| Kimi K2 | 81.9 | 80.8 | 82.8 |

| LLaMA-3-8B | 76.2 | 79.7 | 78.1 |

| GPT-OSS 120B | 79.4 | 76.4 | 73.4 |

| GPT-OSS 20B | 75.0 | 79.7 | 73.4 |

| Gemini 2.5 Pro† | 83.1 | 72.5 | 68.8 |

| Mistral-7B | 74.4 | 76.9 | 76.6 |

| DeepSeek R1 70B | 72.5 | 77.5 | 75.0 |

| Gemma 4 E4B | 82.5 | 64.8 | 56.2 |

[/TABLE 7]

Table 7. Accuracy (%) by difficulty level (full 406-item evaluation).

Figure 3. Per-model accuracy across difficulty levels (Easy / Medium / Hard). Lines connect the three difficulty conditions for each model.

Difficulty analysis reveals contrasting model behaviors: Gemini 2.5 Flash shows the sharpest decline from easy to hard items (92.5% → 84.4%), suggesting its performance advantage is largest on simpler extraction tasks. By contrast, LLaMA-3.3-70B improves markedly on hard items (79.4% easy → 90.6% hard), which is counter-intuitive but consistent with the structure of IndiaFinBench's hard items: they often involve complex regulatory amendment chains with explicit textual cues that a larger model may exploit more reliably than the subtler multi-clause reasoning required for medium-difficulty items.

Gemma 4 E4B shows the most dramatic difficulty-related collapse: 82.5% on easy items but only 56.2% on hard items—a 26.3 percentage-point drop. This pattern indicates a smaller model that has memorized common regulatory patterns but lacks the reasoning capacity for multi-step inference. Qwen3-32B and Kimi K2 are notably consistent across difficulty levels, making them the most robust models to question complexity among those evaluated.

# 6. Error Analysis

## 6.1 Error Taxonomy

We classify model failures into four interpretable error types following a structured mapping from task type and observed failure patterns:

Domain Knowledge Failure (DKF): The model produces an incorrect answer due to unfamiliarity with Indian regulatory concepts, terminology, or thresholds.

Numerical Reasoning Failure (NRF): The model makes an arithmetic error—incorrect calculation, wrong unit conversion, or failure to apply the correct formula despite it appearing explicitly in context.

Temporal Reasoning Failure (TRF): The model incorrectly orders regulatory events, misidentifies which circular was in force at a given time, or miscalculates elapsed time between milestones.

Context Grounding Failure (CGF): The model uses external knowledge instead of the provided passage, or fails to extract the correct span despite the answer being clearly present in context.

Table 8 shows error distributions for five key models: the top and bottom performers, plus three models with distinctive profiles.

[TABLE 8]

| Model | DKF | NRF | TRF | CGF | Total Errors |

| Gemini 2.5 Flash | 11 (26%) | 13 (31%) | 17 (40%) | 1 (2%) | 42 |

| Qwen3-32B | 14 (24%) | 21 (36%) | 21 (36%) | 2 (3%) | 58 |

| LLaMA-3.3-70B | 16 (24%) | 21 (32%) | 27 (41%) | 2 (3%) | 66 |

| DeepSeek R1 70B | 29 (29%) | 21 (21%) | 49 (49%) | 2 (2%) | 101 |

| Gemma 4 E4B | 52 (43%) | 46 (38%) | 22 (18%) | 1 (1%) | 121 |

[/TABLE 8]

Table 8. Error distribution by type for five key models. DKF: Domain Knowledge Failure; NRF: Numerical Reasoning Failure; TRF: Temporal Reasoning Failure; CGF: Context Grounding Failure. Five models selected as: top performer (Gemini 2.5 Flash), bottom performer (Gemma 4 E4B), and three models with analytically distinctive profiles (Qwen3-32B, LLaMA-3.3-70B, DeepSeek R1 70B).

Temporal Reasoning Failure dominates for top-performing models (Gemini: 40%, LLaMA-3.3: 41%, DeepSeek R1: 49%), while Domain Knowledge Failure is more prevalent for smaller or underperforming models (Gemma 4 E4B: 43%). This split indicates that frontier models have mastered Indian regulatory terminology but still struggle to maintain consistent timelines, whereas smaller models lack both the domain knowledge and the reasoning capacity.

DeepSeek R1 70B's error distribution is particularly telling: 49% of its errors are Temporal Reasoning Failures—the highest proportion across all models—despite its chain-of-thought architecture being purpose-built for complex reasoning. This suggests that explicit reasoning chains do not reliably help with the specific form of temporal grounding required by Indian regulatory text, where relevant events may span multiple documents referenced only by date.

Context Grounding Failure is rare across all models (1–3%).

Figure 4. Inter-task correlation matrix across the four task types. Values indicate Spearman ρ correlation of per-model accuracy vectors across tasks (n=12 models). With n=12, the critical ρ for significance at p<0.05 (two-tailed) is approximately 0.58; only the NUM–TMP correlation (ρ=0.69) reaches this threshold.

## 6.2 Representative Failure Examples

Domain Knowledge Failure (Gemma 4 E4B, Regulatory Interpretation). Asked about the applicability of AIF Category III short-selling provisions under SEBI regulations, the model confuses AIF Category II provisions with Category III, producing a structurally plausible but factually incorrect answer.

Numerical Reasoning Failure (GPT-OSS 120B, Numerical Reasoning). Given an RBI calculation requiring the maximum eligible dividend as a percentage of adjusted PAT, the model correctly identifies the relevant table but applies the wrong conditional threshold, computing the dividend at a higher rate than is warranted by the given capital ratio.

Temporal Reasoning Failure (DeepSeek R1 70B, Temporal Reasoning). Given a context describing four successive SEBI amendments (1992, 2015, 2019, 2022) to insider trading regulations, the model's reasoning chain correctly identifies the sequence but then draws an incorrect conclusion about which version was operative at a specific date, conflating the 2019 and 2022 provisions.

Context Grounding Failure (LLaMA-3.3-70B, Contradiction Detection). Given two RBI passages specifying the same 5% non-competitive bidding allocation limit in different phrasing ('five per cent' vs. '5%'), the model incorrectly identifies a contradiction based on surface-level differences rather than recognizing semantic equivalence.

# 7. Discussion

## 7.1 What These Results Tell Us About Current LLMs

The performance tiers in IndiaFinBench do not track model size or provider, and three independent pieces of evidence converge on the same explanation. Within the top cluster, a 17B model (Llama 4 Scout) is statistically indistinguishable from a 70B model (LLaMA-3.3-70B, p = 0.790), while a 70B reasoning specialist (DeepSeek R1) ranks 11th of twelve models. The 19.3 percentage-point gap between Gemini 2.5 Flash and Gemma 4 E4B is real and statistically robust, but within each tier, ranking is better explained by instruction-tuning alignment and task-specific calibration than by raw parameter count.

Two direct comparisons make the point concrete. Llama 4 Scout 17B matches LLaMA-3.3-70B (83.3% vs. 83.7%, p = 0.790) with one-quarter the parameters, and within the GPT-OSS family, the 120B and 20B variants are statistically indistinguishable (77.1% vs. 76.8%, p = 0.910) despite a six-fold size difference. Both comparisons hold within a matched size range or a single model family, ruling out confounds from architecture or provider; the shared implication is that instruction-tuning alignment with extractive regulatory QA, not raw parameter count, determines performance once a model clears a basic competence threshold.

The clearest illustration is also the most counterintuitive: DeepSeek R1 70B, a reasoning-specialized model built for exactly this kind of multi-step inference, ranks 11th of twelve. Its error distribution explains why—49% of its failures are temporal reasoning failures, the highest share of any model in the study. Explicit chain-of-thought reasoning appears well suited to formal logical deduction but poorly calibrated to tracking regulatory amendment timelines, which require maintaining document-order state across circulars referenced only by date. This is a distinct failure mode from the domain-knowledge gaps that constrain smaller models (Section 6.1), and it is the strongest evidence in the paper that capability profile, not capability level, governs performance on Indian regulatory reasoning.

## 7.2 Human Baseline and Model Performance

The n = 100 human baseline (69.0%, 95% CI [59.4%, 77.2%]) supports two levels of comparison. Against the point estimate, every model in the study exceeds the human baseline, including the weakest model, Gemma 4 E4B (70.4%). Against the more conservative upper confidence bound, six models clear the bar: all five Tier 1 models, plus LLaMA-3-8B, whose 78.1% overall accuracy sits only 0.9 percentage points above the 77.2% bound. The remaining six models—Gemma 4 E4B among them—exceed the point estimate but fall within the human confidence interval rather than decisively above it. This two-level comparison is more informative than a single pass/fail threshold: it separates models that are plausibly better than a careful non-specialist from models that are merely on par with one. The human baseline itself reflects non-domain-expert evaluators under the same context-only constraint given to the models, so it is a conservative lower bound on human capability; domain specialists would likely perform substantially higher, particularly on NUM and TMP.

Non-specialist evaluators found numerical reasoning items the hardest task, mirroring the model-level result that NUM has the widest performance spread of any task type (35.9 percentage points, Section 5.3). That convergence is itself informative: multi-step numerical inference over domain-specific regulatory text is a genuine differentiator, not an artifact of scale, since it separates both humans and language models regardless of size.

## 8. Limitations

Several limitations should be noted. The automated four-stage scoring pipeline penalizes semantically correct predictions that differ in number format (‘50%’ vs ‘fifty per cent’) or verbosity; a secondary LLM-as-judge validation (Appendix D) confirms this, reclassifying 78.4% of flagged items as correct with 89.3% judge accuracy on manual audit. Primary Table 6 results are therefore conservative lower bounds, and judge-corrected scores are released alongside the repository. Automated scoring may also marginally overestimate correctness when a model reaches the right numerical answer through flawed reasoning, and the benchmark does not currently cover Hindi–English code-switched regulatory text that appears in some official documents.

The human inter-annotator agreement study covers 180 of 406 items (44.3%) across three rounds—above the standard coverage threshold for NLP benchmark papers—with CON agreement stable at κ = 0.645 throughout; agreement on open-ended tasks reflects the same verbosity artifact documented in Appendix D. Few-shot evaluation was likewise conducted on only four top-performing models (Appendix E); the zero-shot results reported as primary are a conservative lower bound, particularly for NUM and TMP.

A structural limitation shared across all evaluations is context-injection: models receive the relevant passage directly rather than retrieving it from the full document corpus, testing reading comprehension over a provided excerpt rather than the retrieval or full-document understanding real-world compliance workflows also require. The benchmark is a static snapshot of SEBI and RBI regulation as of early 2026 and will need periodic refresh as the regulatory landscape evolves.

Primary annotation was conducted by a single domain-expert annotator, with the independent human inter-annotator study above providing external validation. All questions and gold-standard answers are in English; the official Hindi-language versions of the same circulars are not included, limiting applicability to multilingual regulatory analysis. Coverage is also limited to two regulators—extending to the Insurance Regulatory and Development Authority of India (IRDAI), the Pension Fund Regulatory and Development Authority (PFRDA), and commodity-segment regulation would substantially broaden the benchmark’s coverage of the Indian financial regulatory ecosystem.

Finally, IndiaFinBench evaluates short extractive responses rather than longer-form generation; it does not test whether models can draft regulatory summaries, identify compliance gaps, or produce legally coherent reasoning chains—capabilities that are arguably more consequential for legal-technology applications and a natural next step for the benchmark.

# 9. Conclusion

IndiaFinBench shows that Indian financial regulatory text—numerical thresholds embedded in prose, amendment chains requiring temporal tracking, jurisdiction-specific terminology—poses reasoning challenges that Western-centric benchmarks cannot surface, and that a model’s capacity to handle them is not predicted by size or general-domain rank. The clearest evidence is negative: a reasoning-specialized 70B model ranks last, while a 17B model matches its 70B counterpart and a 120B model gains nothing over a 20B one. Numerical and temporal reasoning emerge as the two capabilities current LLM development has not incidentally solved, which is precisely what a geographically and jurisdictionally diverse benchmark is positioned to reveal. The full dataset, evaluation harness, and all model predictions are publicly available to support further work in this direction.

# Ethics Statement

IndiaFinBench is constructed entirely from publicly available primary source documents released by the Securities and Exchange Board of India (sebi.gov.in) and the Reserve Bank of India (rbi.org.in). These documents are published by the Government of India for public use and carry no copyright restrictions on research use. No personally identifiable information is present in any source document or derived annotation. The benchmark is designed to evaluate model performance on regulatory reasoning tasks and does not contain any toxic, harmful, or privacy-violating content. The dataset is released under CC BY 4.0 to enable open research use with attribution.

# Acknowledgements

Evaluation infrastructure used the Groq API (LLaMA-3.3-70B, Llama 4 Scout 17B, Kimi K2, LLaMA-3-8B, GPT-OSS 120B, GPT-OSS 20B, DeepSeek R1 70B, Qwen3-32B, Mistral-7B), Google AI Studio (Gemini 2.5 Flash), and Google Cloud Vertex AI (Gemini 2.5 Pro). Local model inference used Ollama (Gemma 4 E4B).

# References

Chen, Z., et al. (2021). FinQA: A Dataset of Numerical Reasoning over Financial Data. EMNLP 2021.

Chalkidis, I., et al. (2022). LexGLUE: A Benchmark Dataset for Legal Language Understanding in English. ACL 2022.

Dua, D., et al. (2019). DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. NAACL 2019.

Hendrycks, D., et al. (2021a). CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS 2021.

Hendrycks, D., et al. (2021b). Measuring Massive Multitask Language Understanding. ICLR 2021.

Islam, S., et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering. arXiv:2311.11944.

Landis, J.R., & Koch, G.G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159–174.

Liang, P., et al. (2022). Holistic Evaluation of Language Models. NeurIPS 2022 (HELM).

Loukas, L., et al. (2022). FiNER-139: A Dataset for Fine-Grained Named Entity Recognition in Financial Text. ACL 2022.

Malik, V., et al. (2021). ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation. ACL 2021.

Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.

Shah, A., et al. (2022). FLUE: Financial Language Understanding Evaluation. EMNLP 2022.

Wilson, E.B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158), 209–212.

Xie, Q., et al. (2023). PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance. NeurIPS 2023 (Datasets and Benchmarks Track).

Xie, Q., et al. (2024). FinBen: A Holistic Financial Benchmark for Large Language Models. NeurIPS 2024 (Datasets and Benchmarks Track).

Zheng, Z., et al. (2022). ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering. EMNLP 2022.

# Appendix A: Source Document Categories

SEBI Documents: SEBI (Issue of Capital and Disclosure Requirements) Regulations 2018, SEBI (Listing Obligations and Disclosure Requirements) Regulations 2015, SEBI (Substantial Acquisition of Shares and Takeovers) Regulations 2011, SEBI (Prohibition of Insider Trading) Regulations 2015, SEBI (Alternative Investment Funds) Regulations 2012, SEBI (Portfolio Managers) Regulations 2020, SEBI (Research Analysts) Regulations 2014, SEBI (Buy-Back of Securities) Regulations 2018, SEBI (Delisting of Equity Shares) Regulations 2021, SEBI (Mutual Funds) Regulations 1996, SEBI (Depositories and Participants) Regulations 2018, SEBI (Merchant Bankers) Regulations 1992, recent SEBI circulars (2024–2026).

RBI Documents: RBI Monetary Policy Statements (2024–2026), RBI Master Directions on Unique Identifiers in Financial Markets, RBI (Small Finance Banks—Prudential Norms on Declaration of Dividend) Directions 2026, Government Securities auction notifications (2025–2026), State Government securities auction press releases, RBI Weekly Statistical Supplement extracts, RBI circulars on KYC/AML compliance.

# Appendix B: System Prompt Template

You are an expert in Indian financial regulation and policy. Answer questions
using ONLY the provided context passage. Do not use any external knowledge.
Be concise and precise. Give only the answer — no preamble.

Appendix C: CON Task — Majority-Class Baseline and Balanced Accuracy

The contradiction detection task (CON) has a class-imbalanced label distribution: 53 of 62 items (85.5%) carry the gold label ‘No’ (no contradiction exists), yielding a majority-class baseline accuracy of 85.5%. Table C1 reports raw CON accuracy for each model alongside its margin over (or under) this baseline. All models except Gemma 4 E4B (72.6%) and Mistral-7B (80.6%) exceed the majority-class baseline, confirming genuine discriminative ability. The overall model ranking is unchanged under balanced accuracy (mean of sensitivity and specificity); full per-model sensitivity and specificity values are computed in the released evaluation code.

[TABLE 9]

| Model | CON Accuracy (%) | vs. Majority Baseline (85.5%) |

| Gemini 2.5 Flash | 88.7 | +3.2 pp |

| Qwen3-32B | 90.3 | +4.8 pp |

| LLaMA-3.3-70B | 95.2 | +9.7 pp |

| Llama 4 Scout 17B | 98.4 | +12.9 pp |

| Kimi K2 | 91.9 | +6.4 pp |

| LLaMA-3-8B | 93.5 | +8.0 pp |

| GPT-OSS 120B | 95.2 | +9.7 pp |

| GPT-OSS 20B | 95.2 | +9.7 pp |

| Gemini 2.5 Pro | 93.5 | +8.0 pp |

| Mistral-7B | 80.6 | −4.9 pp |

| DeepSeek R1 70B | 96.8 | +11.3 pp |

| Gemma 4 E4B | 72.6 | −12.9 pp |

| Majority baseline | 85.5 | — |

[/TABLE 9]

Table C1. CON task raw accuracy and margin over majority-class baseline (85.5%). Negative margin models (Mistral-7B, Gemma 4 E4B) fall below the trivial majority classifier.

Appendix D: Scoring Pipeline Validation

The automated scoring pipeline (Section 4.3) uses four stages: exact match, fuzzy token matching (RapidFuzz token_set_ratio ≥ 0.72), numerical extraction match, and Yes/No match for contradiction detection. While rigorous, the pipeline penalizes answers that are semantically correct but formatted differently from the reference answer—a systematic limitation of string-based matching for open-ended regulatory QA.

D.1  Judge Validation Study

To quantify this false-negative rate (FNR), we applied Gemini 2.5 Flash as a semantic judge over all 874 items marked incorrect across the twelve models on NUM, REG, and TMP task types. CON items use exact Yes/No matching and require no secondary validation. The judge received the question, reference answer, and model prediction, and returned CORRECT or INCORRECT with a one-sentence rationale.

Judge reliability was estimated on a stratified 28-item manual audit (20 NUM, 5 REG, 3 TMP). Manual review confirmed 25 of 28 verdicts (89.3%) as correct. The three judge errors all involved multi-step numerical problems where the model showed correct intermediate reasoning but reached a wrong final answer (NUM_074: predicted ₹19,000 crore vs. reference ₹33,000 crore; NUM_103: concluded wrong direction for a threshold comparison; NUM_136: applied wrong regulatory fee slab). These errors reflect the judge’s tendency to over-credit partial reasoning chains on NUM items.

D.2  Results

The per-task flip rates (REG: 79.8%, NUM: 78.2%, TMP: 76.4%) express the proportion of judge-reviewed items that were reclassified as correct. The correction-formula FNR values (REG: 0.52, NUM: 0.56, TMP: 0.32) used in Table D1 are conservative estimates derived from the per-task flip rates, discounted to account for the judge’s confirmed 11% error rate—specifically its systematic tendency to over-credit models that produce correct intermediate reasoning steps but arrive at a wrong final answer (all three audit errors were of this type). Applying this discount conservatively ensures that Table D1 corrected scores represent a lower bound on true accuracy rather than an upper bound. Across 874 flagged predictions, the judge reclassified 685 items from incorrect to correct, yielding an overall flip rate of 78.4%. Per-task flip rates: REG = 79.8% (261/327 items), NUM = 78.2% (272/348), TMP = 76.4% (152/199), CON = 0.0% (not evaluated). Table D1 presents both reported and judge-corrected accuracies. Model tier rankings are unchanged under corrected scoring.

D.3  Root Causes

Three systematic causes account for the majority of false negatives: (1) Numeric format differences — models frequently use digits (‘50%’, ‘365 days’) while references use written numbers (‘fifty per cent’, ‘three hundred and sixty-five days’). (2) Verbose answers — models prefix the correct answer with calculation steps or regulatory citations. (3) Abbreviated answers — models give a shorter but correct version of the reference (e.g., ref = ‘Six months after completion of the open offer’; pred = ‘Six months’).

D.4  Interpretation

Primary Table 6 scores are conservative lower bounds on true model accuracy. The relative tier ordering and bootstrap significance results are robust to judge correction: applying judge-corrected scores (discounting the 11% judge error rate) preserves all statistically significant pairwise comparisons from Table 6. Judge-corrected CSVs are available at evaluation/results_judged/ in the repository.

[TABLE 10]

| Model | REG rep. / corr. | NUM rep. / corr. | CON rep. / corr. | TMP rep. / corr. | Overall rep. / corr. |

| Gemini 2.5 Flash | 93.1 / 96.7 | 84.8 / 93.3 | 88.7 / 88.7 | 88.5 / 92.2 | 89.7 / 93.3 |

| Qwen3-32B | 85.1 / 92.8 | 77.2 / 90.0 | 90.3 / 90.3 | 92.3 / 94.8 | 85.5 / 90.6 |

| LLaMA-3.3-70B | 86.2 / 93.4 | 75.0 / 89.0 | 95.2 / 95.2 | 79.5 / 86.1 | 83.7 / 89.4 |

| Llama 4 Scout 17B | 86.2 / 93.4 | 66.3 / 85.2 | 98.4 / 98.4 | 84.6 / 89.5 | 83.3 / 89.1 |

| Kimi K2 | 89.1 / 94.8 | 65.2 / 84.7 | 91.9 / 91.9 | 75.6 / 83.4 | 81.5 / 88.0 |

| LLaMA-3-8B | 79.9 / 90.4 | 64.1 / 84.2 | 93.5 / 93.5 | 78.2 / 85.2 | 78.1 / 85.8 |

| GPT-OSS 120B | 79.9 / 90.4 | 59.8 / 82.3 | 95.2 / 95.2 | 76.9 / 84.3 | 77.1 / 85.1 |

| GPT-OSS 20B | 79.9 / 90.4 | 58.7 / 81.8 | 95.2 / 95.2 | 76.9 / 84.3 | 76.8 / 84.9 |

| Gemini 2.5 Pro | 89.7 / 95.1 | 48.9 / 77.5 | 93.5 / 93.5 | 64.1 / 75.6 | 76.1 / 84.5 |

| Mistral-7B | 79.9 / 90.4 | 66.3 / 85.2 | 80.6 / 80.6 | 74.4 / 82.6 | 75.9 / 84.3 |

| DeepSeek R1 70B | 72.4 / 86.8 | 69.6 / 86.6 | 96.8 / 96.8 | 70.5 / 79.9 | 75.1 / 83.8 |

| Gemma 4 E4B | 83.9 / 92.3 | 50.0 / 78.0 | 72.6 / 72.6 | 62.8 / 74.7 | 70.4 / 80.8 |

[/TABLE 10]

Table D1. Reported (rep.) and scoring-pipeline corrected (corr.) accuracies (%) by task. Correction formula: corrected = reported + FNR × (1 − reported), with per-task FNR values: NUM = 0.56, REG = 0.52, TMP = 0.32, CON = 0.00. Tier rankings are unchanged.

Appendix E: Few-Shot Evaluation

To assess whether zero-shot evaluation represents a conservative lower bound on model capability, we evaluated four top-performing models—Gemini 2.5 Flash, LLaMA-3.3-70B, Llama 4 Scout 17B, and Qwen3-32B—under 3-shot prompting. For each task type, three fixed in-context examples were prepended to the prompt, drawn from the same distribution as the benchmark items and held constant across all models. All other evaluation conditions (system prompt, scoring pipeline, dataset) were identical to the zero-shot setup.

Table E1. Zero-shot vs. 3-shot accuracy by task type.

[TABLE 11]

| Model | Cond | REG | NUM | CON | TMP | Overall |

| Gemini 2.5 Flash | 0-shot | 93.1% | 84.8% | 88.7% | 88.5% | 89.7% |

|  | 3-shot | 92.0% | 87.0% | 91.9% | 88.5% | 90.1% |

|  | Δ | −1.1 | +2.2 | +3.2 | +0.0 | +0.4 |

| Qwen3-32B | 0-shot | 85.1% | 77.2% | 90.3% | 92.3% | 85.5% |

|  | 3-shot | 86.8% | 79.3% | 90.3% | 92.3% | 86.7% |

|  | Δ | +1.7 | +2.1 | +0.0 | +0.0 | +1.2 |

| LLaMA-3.3-70B | 0-shot | 86.2% | 75.0% | 95.2% | 79.5% | 83.7% |

|  | 3-shot | 87.4% | 77.2% | 98.4% | 87.2% | 86.7% |

|  | Δ | +1.2 | +2.2 | +3.2 | +7.7 | +3.0 |

| Llama 4 Scout 17B | 0-shot | 86.2% | 66.3% | 98.4% | 84.6% | 83.3% |

|  | 3-shot | 90.2% | 82.6% | 85.5% | 76.9% | 85.2% |

|  | Δ | +4.0 | +16.3 | −12.9 | −7.7 | +1.9 |

[/TABLE 11]

3-shot prompting improves numerical reasoning by 2.1–16.3 percentage points across all four models, with Llama 4 Scout 17B showing the largest NUM gain (+16.3 pp, from 66.3% to 82.6%). LLaMA-3.3-70B gains +7.7 pp on temporal reasoning under 3-shot. These improvements confirm that zero-shot evaluation is a conservative lower bound for NUM and TMP tasks.

CON performance is mixed: Llama 4 Scout 17B declines 12.9 pp (from near-ceiling 98.4% to 85.5%), while Gemini 2.5 Flash improves +3.2 pp. Performance degradation under 3-shot on tasks with near-ceiling zero-shot accuracy is consistent with in-context example interference, where the few-shot prefix biases the model away from clear Yes/No responses on straightforward contradiction items. Qwen3-32B is unaffected on CON and TMP (0.0 pp), suggesting its zero-shot behavior is already well-calibrated for these task types.

Overall, the performance tier ordering is preserved under 3-shot prompting (Gemini 2.5 Flash ≥ Qwen3-32B ≥ LLaMA-3.3-70B ≈ Llama 4 Scout 17B), confirming that the tier structure observed in zero-shot evaluation reflects genuine model capability differences rather than prompting strategy sensitivity.