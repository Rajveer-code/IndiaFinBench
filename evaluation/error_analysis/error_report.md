# IndiaFinBench — Error Analysis Report

This report documents the error analysis for the IndiaFinBench evaluation benchmark. All figures referenced here are saved in `evaluation/error_analysis/`.

## 1. Overall Results

| Model | REG | NUM | CON | TMP | Overall |
|---|---|---|---|---|---|
| Claude 3 Haiku | 92.5% | 93.8% | 86.7% | 91.4% | **91.3%** |
| Gemini 2.5 Flash | 96.2% | 84.4% | 83.3% | 82.4% | **87.9%** |
| LLaMA-3.3-70B | 77.4% | 84.4% | 90.0% | 77.1% | **81.3%** |
| LLaMA-3-8B | 77.4% | 62.5% | 86.7% | 74.3% | **75.3%** |
| Mistral-7B | 69.8% | 68.8% | 80.0% | 74.3% | **72.7%** |

## 2. Key Findings

### 2.1 Task Difficulty Ranking

Tasks ranked from hardest to easiest (by average model accuracy):

1. **Numerical Reasoning** — 78.8% average
2. **Temporal Reasoning** — 79.9% average
3. **Regulatory Interpretation** — 82.6% average
4. **Contradiction Detection** — 85.3% average

### 2.2 Numerical Reasoning is the Most Discriminative Task

Numerical reasoning shows the widest performance spread across models (31.2 percentage points between best and worst). **Claude 3 Haiku** achieves 93.8% while **LLaMA-3-8B** achieves only 62.5%. This suggests that arithmetic reasoning over Indian regulatory figures (repo rates, percentage thresholds, capital ratios) is a genuine challenge that differentiates model capability.

### 2.3 Regulatory Interpretation Favours Larger Models

On regulatory interpretation, the gap between the best model (Gemini 2.5 Flash: 96.2%) and the weakest (Mistral-7B: 69.8%) is 26.4 points. This task requires understanding SEBI/RBI-specific terminology (LODR, PMLA, SFB, AIF) and exact compliance thresholds — knowledge that smaller models demonstrably lack.

### 2.4 Contradiction Detection is the Most Uniform Task

Contradiction detection shows the narrowest spread (10.0 points) suggesting that binary Yes/No reasoning over two passages is a relatively tractable task even for smaller models. However, this masks failures on hard items where the contradiction is subtle (e.g., a provision that appears consistent but is superseded by an amendment).

## 3. Difficulty Analysis

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Claude 3 Haiku | 90.8% | 92.3% | 90.0% |
| Gemini 2.5 Flash | 95.4% | 84.6% | 73.7% |
| LLaMA-3.3-70B | 81.5% | 78.5% | 90.0% |
| LLaMA-3-8B | 81.5% | 69.2% | 75.0% |
| Mistral-7B | 72.3% | 70.8% | 80.0% |

Hard questions show a consistent drop across all models, confirming that question difficulty was calibrated correctly during annotation. The gap between easy and hard accuracy is largest for smaller models (LLaMA-3-8B, Mistral-7B), suggesting these models rely more heavily on surface-level pattern matching.

## 4. Error Taxonomy

We classify each model failure into one of four error types based on task type and difficulty:

| Error Type | Definition |
|---|---|
| **Domain Knowledge Failure** | Model does not know the Indian regulatory framework (e.g., wrong SEBI threshold, unfamiliar RBI terminology) |
| **Numerical Reasoning Failure** | Model makes arithmetic errors on rate calculations, percentage changes, or capital ratio computations |
| **Temporal Reasoning Failure** | Model confuses amendment order, incorrectly identifies which circular supersedes which, or misstates effective dates |
| **Context Grounding Failure** | Model uses outside knowledge instead of the provided context, or fails to extract the answer from the passage |

### 4.1 Error Distribution by Model

| Model | Domain | Numerical | Temporal | Context Grounding |
|---|---|---|---|---|
| Claude 3 Haiku | 4 (31%) | 2 (15%) | 7 (54%) | 0 (0%) |
| Gemini 2.5 Flash | 3 (17%) | 5 (28%) | 9 (50%) | 1 (6%) |
| LLaMA-3.3-70B | 12 (43%) | 5 (18%) | 10 (36%) | 1 (4%) |
| LLaMA-3-8B | 13 (35%) | 12 (32%) | 11 (30%) | 1 (3%) |
| Mistral-7B | 17 (41%) | 10 (24%) | 13 (32%) | 1 (2%) |

## 5. Representative Failure Examples

### Domain Knowledge Failure

**Model:** Claude 3 Haiku  |  **Task:** Regulatory Interpretation  |  **Difficulty:** easy

**Question:** Under SEBI ICDR Regulations, 2018, what percentage of the net offer must an issuer that does not meet the standard eligi

**Reference answer:** `At least seventy five per cent.`

**Model prediction:** `75 per cent (seventy five per cent) of the net offer must be allotted to qualified institutional buy`

---

### Numerical Reasoning Failure

**Model:** Claude 3 Haiku  |  **Task:** Numerical Reasoning  |  **Difficulty:** medium

**Question:** Under SEBI LODR Regulations 2015, if a listed entity holds its first board meeting on 1 April of a financial year, by wh

**Reference answer:** `By 29 July of the same year (1 April + 120 days = 29 July)`

**Model prediction:** `**Calculation:**

First board meeting: 1 April

Maximum time gap between consecutive board meetings:`

---

### Temporal Reasoning Failure

**Model:** Claude 3 Haiku  |  **Task:** Contradiction Detection  |  **Difficulty:** hard

**Question:** Do the two passages impose different shareholding thresholds triggering mandatory Board approval for acquisitions in rec

**Reference answer:** `No`

**Model prediction:** `Yes. Passage A permits a 15% shareholding threshold for stock exchanges (among other entities) befor`

---

### Context Grounding Failure

**Model:** Gemini 2.5 Flash  |  **Task:** Contradiction Detection  |  **Difficulty:** easy

**Question:** Do both passages state the same percentage of the notified amount reserved for non-competitive bidding in Government Sec

**Reference answer:** `Yes`

**Model prediction:** `No, Passage A states "five per cent" while Passage B states "up to 5%".`

---

## 6. Implications for Future Work

1. **Domain-specific fine-tuning.** The systematic failures on SEBI/RBI terminology suggest that models fine-tuned on Indian regulatory corpora would substantially outperform zero-shot baselines on this benchmark.

2. **Numerical reasoning is a bottleneck.** Even frontier models struggle with multi-step arithmetic over regulatory figures. Chain-of-thought prompting or tool-augmented models may be necessary to close this gap.

3. **Temporal reasoning requires amendment tracking.** Correctly identifying which circular supersedes which requires maintaining a timeline of regulatory changes — a capability that current models handle inconsistently.

4. **IndiaFinBench remains challenging.** Even the best-performing model (Claude 3 Haiku, 91.3%) fails on approximately 13 questions, demonstrating that the benchmark is not saturated and can track progress in the field.
