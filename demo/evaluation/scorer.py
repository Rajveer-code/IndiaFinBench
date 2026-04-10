"""
evaluation/scorer.py
--------------------
Purpose:  Scoring functions for IndiaFinBench submissions.
          Implements exact match, partial-credit F1 (SQuAD-style), and the
          task-aware composite score used in the leaderboard.
Inputs:   gold_answer (str), prediction (str), task_type (str)
Outputs:  score (float, 0.0–1.0)
Usage:
    from evaluation.scorer import score_submission
    result = score_submission(predictions, questions)
"""

import re
from collections import Counter
from typing import Any

try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


# ── Normalisation ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[^\w\s%.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _extract_yn(text: str) -> str:
    """Extract Yes/No label from start of text."""
    t = _normalise(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "unclear"


# ── Exact match ────────────────────────────────────────────────────────────────

def exact_match(gold: str, pred: str) -> float:
    """Case-insensitive exact match after normalisation.

    Args:
        gold: Reference answer string.
        pred: Predicted answer string.

    Returns:
        1.0 if match, 0.0 otherwise.
    """
    return 1.0 if _normalise(gold) == _normalise(pred) else 0.0


def exact_match_with_variants(
    gold: str,
    variants: list[str],
    pred: str,
) -> float:
    """Exact match against gold answer and all its variants.

    Args:
        gold:     Primary reference answer.
        variants: List of acceptable alternative answers.
        pred:     Predicted answer.

    Returns:
        1.0 if pred matches any acceptable answer, 0.0 otherwise.
    """
    all_refs = [gold] + (variants or [])
    pn = _normalise(pred)
    return 1.0 if any(_normalise(r) == pn for r in all_refs) else 0.0


# ── Partial credit F1 (SQuAD-style) ───────────────────────────────────────────

def partial_credit_f1(gold: str, pred: str) -> float:
    """Token-overlap F1, same formula as SQuAD evaluation script.

    Args:
        gold: Reference answer string.
        pred: Predicted answer string.

    Returns:
        F1 score in [0.0, 1.0].
    """
    gold_tokens = _normalise(gold).split()
    pred_tokens = _normalise(pred).split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(gold_tokens) & Counter(pred_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Fuzzy match ────────────────────────────────────────────────────────────────

def fuzzy_match(gold: str, pred: str, threshold: float = 0.72) -> float:
    """Token-set-ratio fuzzy match using rapidfuzz (falls back to exact).

    Args:
        gold:      Reference answer string.
        pred:      Predicted answer string.
        threshold: Minimum ratio to count as a match (0–1).

    Returns:
        1.0 if fuzzy ratio >= threshold, 0.0 otherwise.
    """
    gn, pn = _normalise(gold), _normalise(pred)
    if not gn or not pn:
        return 0.0
    if gn == pn:
        return 1.0

    # Numeric agreement shortcut
    nums_g = set(re.findall(r"\d[\d,]*\.?\d*", gn))
    nums_p = set(re.findall(r"\d[\d,]*\.?\d*", pn))
    if nums_g and nums_p and nums_g == nums_p:
        return 1.0

    if _HAS_RAPIDFUZZ:
        ratio = _rfuzz.token_set_ratio(gn, pn) / 100.0
        return 1.0 if ratio >= threshold else 0.0

    # Fallback to F1
    return 1.0 if partial_credit_f1(gn, pn) >= threshold else 0.0


# ── Task-aware item scorer ─────────────────────────────────────────────────────

def score_item(
    gold: str,
    variants: list[str],
    pred: str,
    task_type: str,
) -> float:
    """Score a single prediction against the gold answer for a given task.

    Scoring rules:
      - contradiction_detection: exact Yes/No match (binary).
      - numerical_reasoning:     numeric-aware fuzzy match (strict).
      - regulatory_interpretation / temporal_reasoning: fuzzy match (threshold 0.72).

    Args:
        gold:      Gold answer string.
        variants:  List of acceptable alternative answers.
        pred:      Model prediction string.
        task_type: One of the 4 IndiaFinBench task type strings.

    Returns:
        Score in [0.0, 1.0].
    """
    if not pred or "fail:" in pred.lower() or "cannot be determined" in pred.lower():
        return 0.0

    if task_type == "contradiction_detection":
        ref_yn  = _extract_yn(gold)
        pred_yn = _extract_yn(pred)
        return 1.0 if (ref_yn == pred_yn and ref_yn != "unclear") else 0.0

    # Check variants first (exact match against any variant)
    if exact_match_with_variants(gold, variants, pred) == 1.0:
        return 1.0

    return fuzzy_match(gold, pred)


# ── Batch scorer ───────────────────────────────────────────────────────────────

def score_submission(
    predictions: dict[str, str],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score a full submission against the benchmark questions.

    Args:
        predictions: Dict mapping question id -> model prediction string.
        questions:   List of question dicts (fields: id, task_type, gold_answer,
                     gold_answer_variants).

    Returns:
        Dict with keys:
          - overall:    float, overall accuracy (0–1).
          - per_task:   dict task_short -> float accuracy.
          - per_item:   list of dicts {id, task_type, correct, score}.
          - n_scored:   int, number of items that had a prediction.
          - n_total:    int, total benchmark items.
    """
    TASK_SHORT = {
        "regulatory_interpretation": "REG",
        "numerical_reasoning":       "NUM",
        "contradiction_detection":   "CON",
        "temporal_reasoning":        "TMP",
    }

    task_correct: dict[str, list[float]] = {ts: [] for ts in TASK_SHORT.values()}
    per_item: list[dict] = []
    n_scored = 0

    for q in questions:
        qid      = q["id"]
        task     = q["task_type"]
        gold     = q["gold_answer"]
        variants = q.get("gold_answer_variants", [])
        pred     = predictions.get(qid, "")

        if pred:
            n_scored += 1

        sc = score_item(gold, variants, pred, task)
        ts = TASK_SHORT.get(task, task)
        task_correct[ts].append(sc)
        per_item.append({
            "id":        qid,
            "task_type": task,
            "correct":   int(sc >= 0.5),
            "score":     sc,
        })

    per_task = {
        ts: (sum(scores) / len(scores)) if scores else 0.0
        for ts, scores in task_correct.items()
    }
    all_scores = [item["score"] for item in per_item]
    overall    = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "overall":  overall,
        "per_task": per_task,
        "per_item": per_item,
        "n_scored": n_scored,
        "n_total":  len(questions),
    }
