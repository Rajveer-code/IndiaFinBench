# pip install datasets huggingface_hub pandas scikit-learn
"""
upload_to_huggingface.py
-------------------------
Loads the IndiaFinBench benchmark from the local annotation directory,
converts it to a HuggingFace Dataset, creates stratified train/test splits,
writes a dataset card (README_HF.md), and pushes everything to HuggingFace Hub.

Usage:
    # 1. Install dependencies:
    pip install datasets huggingface_hub pandas scikit-learn

    # 2. Login (one-time):
    huggingface-cli login

    # 3. Run:
    python scripts/upload_to_huggingface.py

Output:
    - HuggingFace dataset at: https://huggingface.co/datasets/Rajveer-code/IndiaFinBench
    - Local dataset card:     README_HF.md
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_PATH        = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
CARD_PATH      = "README_HF.md"
HF_REPO_ID     = "Rajveer-code/IndiaFinBench"
RANDOM_STATE   = 42
TEST_FRACTION  = 0.80   # 120 items
VAL_FRACTION   = 0.20   # 30 items

TASK_TYPE_MAP = {
    "regulatory_interpretation": "regulatory_interpretation",
    "numerical_reasoning":       "numerical_reasoning",
    "contradiction_detection":   "contradiction_detection",
    "temporal_reasoning":        "temporal_reasoning",
}

# ---------------------------------------------------------------------------
# Helper: normalise a raw JSON record into the canonical schema
# ---------------------------------------------------------------------------

def normalise_record(raw: dict, index: int) -> dict:
    """
    Convert a raw JSON item from indiafinbench_qa_combined_150.json into the
    canonical HuggingFace column schema.

    Parameters
    ----------
    raw : dict
        A single record from the raw JSON file.
    index : int
        Zero-based position (used to construct id if missing).

    Returns
    -------
    dict
        Normalised record with exactly the required columns.
    """
    item_id = raw.get("id", f"indiafinbench_{index + 1:03d}")

    # For contradiction detection items, combine context_a + context_b
    if raw.get("task_type") == "contradiction_detection":
        context = (
            f"Passage A:\n{raw.get('context_a', '')}\n\n"
            f"Passage B:\n{raw.get('context_b', '')}"
        )
        source_doc = raw.get(
            "document_a",
            raw.get("document", raw.get("regulation_a", "unknown"))
        )
    else:
        context = raw.get("context", "")
        source_doc = raw.get("document", raw.get("regulation", "unknown"))

    return {
        "id":               f"indiafinbench_{item_id}",
        "task_type":        TASK_TYPE_MAP.get(raw.get("task_type", ""), raw.get("task_type", "")),
        "difficulty":       raw.get("difficulty", "medium").lower(),
        "source":           raw.get("source", "SEBI").upper(),
        "context":          context,
        "question":         raw.get("question", ""),
        "reference_answer": raw.get("answer", ""),
        "source_document":  source_doc,
    }


# ---------------------------------------------------------------------------
# Step 1–4: Load, normalise, split, build DatasetDict
# ---------------------------------------------------------------------------

def build_dataset():
    """
    Load raw QA data, normalise, stratify-split, and return a HuggingFace
    DatasetDict with 'validation' and 'test' keys.

    Returns
    -------
    datasets.DatasetDict
        A dict with keys 'validation' (30 items) and 'test' (120 items),
        stratified by task_type with random_state=42.
    """
    try:
        from datasets import Dataset, DatasetDict
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"\nERROR: Missing dependency — {e}")
        print("Run:  pip install datasets huggingface_hub pandas scikit-learn\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    qa_path = Path(QA_PATH)
    if not qa_path.exists():
        print(f"\nERROR: QA file not found at {QA_PATH}")
        print("Make sure you are running this from the IndiaFinBench/ root.\n")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\nLoaded {len(raw_data)} items from {QA_PATH}")

    # ------------------------------------------------------------------
    # Normalise records
    # ------------------------------------------------------------------
    records = [normalise_record(item, i) for i, item in enumerate(raw_data)]

    # ------------------------------------------------------------------
    # Stratified split: 80% test, 20% validation
    # ------------------------------------------------------------------
    labels = [r["task_type"] for r in records]

    try:
        test_records, val_records = train_test_split(
            records,
            test_size=VAL_FRACTION,
            stratify=labels,
            random_state=RANDOM_STATE,
        )
    except ValueError as e:
        print(f"\nWARNING: Stratified split failed ({e}). Falling back to random split.")
        import random
        random.seed(RANDOM_STATE)
        shuffled = records[:]
        random.shuffle(shuffled)
        n_val = round(len(shuffled) * VAL_FRACTION)
        val_records  = shuffled[:n_val]
        test_records = shuffled[n_val:]

    print(f"Split: {len(test_records)} test / {len(val_records)} validation")
    print(f"Random state: {RANDOM_STATE} (reproducible)")

    # ------------------------------------------------------------------
    # Build DatasetDict
    # ------------------------------------------------------------------
    dataset_dict = DatasetDict({
        "test":       Dataset.from_list(test_records),
        "validation": Dataset.from_list(val_records),
    })

    return dataset_dict


# ---------------------------------------------------------------------------
# Step 5: Write dataset card
# ---------------------------------------------------------------------------

DATASET_CARD = """\
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
"""


def write_dataset_card(path: str = CARD_PATH) -> None:
    """
    Write the HuggingFace dataset card (README_HF.md) to disk.

    Parameters
    ----------
    path : str
        Output file path for the dataset card.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD)
    print(f"Dataset card written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Requires: pip install datasets huggingface_hub pandas scikit-learn
    # Before running: huggingface-cli login

    print("=" * 62)
    print("  IndiaFinBench — HuggingFace Upload Script")
    print("=" * 62)

    # Step 1–4: Build dataset
    dataset_dict = build_dataset()

    # Step 5: Write dataset card
    write_dataset_card(CARD_PATH)

    # Step 6: Push to Hub
    try:
        dataset_dict.push_to_hub(
            HF_REPO_ID,
            private=False,
            token=None,  # uses cached token from huggingface-cli login
        )

        print("\nDataset pushed successfully.")
        print(f"Visit: https://huggingface.co/datasets/{HF_REPO_ID}")

    except Exception as e:
        print(f"\nERROR during push_to_hub: {e}")
        print(
            "\nTroubleshooting:\n"
            "  1. Run 'huggingface-cli login' and enter your token.\n"
            "  2. Ensure you have write access to the repository.\n"
            "  3. Check your internet connection.\n"
        )
        sys.exit(1)
