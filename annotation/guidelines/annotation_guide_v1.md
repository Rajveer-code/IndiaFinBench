# IndiaFinBench — Annotation Guidelines v1.0

## Overview
Each item is a (context, question, answer, task_type) tuple.
Context is a passage extracted from an official SEBI or RBI document.
The annotator's job is to READ the context and ANSWER the question
based solely on what the context says.

## Your Role as Annotator
You are NOT writing questions. You are answering them.
Read the context passage carefully, then write the answer.
Do NOT use any outside knowledge — answer only from the context.

## Task Types

### 1. regulatory_interpretation
Answer requires identifying a rule, deadline, or scope of applicability.
Example answer format: "30 days", "Stock brokers and sub-brokers", "SEBI Board approval"

### 2. numerical_reasoning
Answer requires arithmetic on figures in the document.
Always include the unit in your answer: "6.5%", "0.50 percentage points", "Rs. 25 crore"

### 3. contradiction_detection
Two passages are given. Answer is "Yes" (they contradict) or "No" (they do not).
Follow with one sentence of explanation.

### 4. temporal_reasoning
Answer requires establishing which event/rule came first, or what the
current state is after a series of changes.

## Quality Rules
- Answer must come from the context alone
- If you cannot answer from the context, write: "Cannot determine from context"
- For numerical answers, show your working if a calculation was needed
- Keep answers concise — do not write paragraphs

## Inter-Annotator Agreement
Your answers will be compared with a second annotator's answers.
Items where both annotators agree (Cohen's Kappa ≥ 0.70) are kept.
Items with disagreement are discarded. This is normal and expected.
