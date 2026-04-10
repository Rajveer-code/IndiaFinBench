"""
evaluation/tasks.py
-------------------
Purpose:  Task-specific prompt templates and answer extraction logic for
          IndiaFinBench evaluation.
Inputs:   question dict from questions.json
Outputs:  Prompt string; extracted answer string from raw model output
Usage:
    from evaluation.tasks import build_prompt, extract_answer
"""

import re


SYSTEM_PROMPT = (
    "You are an expert in Indian financial regulation and policy. "
    "Answer questions using ONLY the provided context passage. "
    "Do not use any external knowledge. "
    "Be concise and precise. Give only the answer — no preamble."
)

# Maximum context characters per passage (word-boundary trimming done in build_prompt)
MAX_CTX_WORDS = 450   # ~3000 chars


def _trim(text: str, max_words: int = MAX_CTX_WORDS) -> str:
    """Trim text to at most max_words words (word-boundary safe).

    Args:
        text:      Input string.
        max_words: Maximum number of words to keep.

    Returns:
        Trimmed string.
    """
    words = text.split()
    return " ".join(words[:max_words])


def build_prompt(item: dict) -> str:
    """Build the evaluation prompt for a single benchmark item.

    Args:
        item: Question dict with fields: task_type, context, context_a,
              context_b, question.

    Returns:
        Formatted prompt string ready to pass to a language model.
    """
    task = item["task_type"]
    q    = item["question"]

    if task == "contradiction_detection":
        ctx_a = _trim(item.get("context_a", ""), 200)
        ctx_b = _trim(item.get("context_b", ""), 200)
        return (
            f"Passage A:\n{ctx_a}\n\n"
            f"Passage B:\n{ctx_b}\n\n"
            f"Question: {q}\n\n"
            f"Answer with 'Yes' or 'No' then one sentence of explanation:"
        )

    ctx = _trim(item.get("context", ""))

    if task == "numerical_reasoning":
        return (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Show your calculation and give the final answer with units:"
        )

    if task == "temporal_reasoning":
        return (
            f"Context:\n{ctx}\n\n"
            f"Question: {q}\n\n"
            f"Answer precisely, noting relevant dates or sequences:"
        )

    # regulatory_interpretation
    return f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"


def extract_answer(raw_output: str, task_type: str) -> str:
    """Extract a clean answer from raw model output.

    Strips preamble patterns like "Answer:", "The answer is:", etc.

    Args:
        raw_output: Raw string returned by the model.
        task_type:  Task type for task-specific extraction rules.

    Returns:
        Cleaned answer string.
    """
    if not raw_output:
        return ""

    text = raw_output.strip()

    # Strip common answer prefixes
    preambles = [
        r"(?i)^(the\s+)?answer\s+(is|:)\s*",
        r"(?i)^based on .*?,?\s*",
        r"(?i)^according to .*?,?\s*",
        r"(?i)^from the (context|passage).*?,?\s*",
    ]
    for pat in preambles:
        text = re.sub(pat, "", text, flags=re.MULTILINE).strip()

    # For contradiction_detection: keep only first sentence
    if task_type == "contradiction_detection":
        # First word should be Yes/No
        first_line = text.split("\n")[0].strip()
        # Take only up to the first period/newline for the Yes/No part
        return first_line

    # Truncate to first 200 chars for extractive tasks
    return text[:200].strip()
