"""
demo/app.py
------------
Gradio demo for IndiaFinBench — deployable to HuggingFace Spaces.

Loads the IndiaFinBench dataset from HuggingFace Hub and provides an
interactive interface to browse random questions by task type and difficulty,
with a hidden reference answer reveal feature.

Usage (local):
    pip install gradio datasets
    python demo/app.py

HuggingFace Spaces:
    Upload this file and requirements.txt to a new Space.
    The Space will auto-install dependencies and launch the app.
"""

import random
from typing import Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASET_REPO = "Rajveer-code/IndiaFinBench"
_dataset_cache: Optional[list] = None


def load_dataset_once() -> list:
    """
    Load and cache the IndiaFinBench test split from HuggingFace Hub.

    Returns
    -------
    list
        List of dataset records (dicts).
    """
    global _dataset_cache
    if _dataset_cache is None:
        try:
            from datasets import load_dataset
            ds = load_dataset(DATASET_REPO, split="test")
            _dataset_cache = list(ds)
        except Exception as e:
            # Graceful fallback: return a placeholder record so the UI doesn't crash
            _dataset_cache = [
                {
                    "id": "placeholder_001",
                    "task_type": "regulatory_interpretation",
                    "difficulty": "easy",
                    "source": "SEBI",
                    "context": (
                        "[Dataset not yet available on HuggingFace. "
                        "Please run scripts/upload_to_huggingface.py first.] "
                        f"Error: {e}"
                    ),
                    "question": "When will IndiaFinBench be available?",
                    "reference_answer": "Upon upload to HuggingFace Hub.",
                    "source_document": "N/A",
                }
            ]
    return _dataset_cache


# ---------------------------------------------------------------------------
# Filtering helper
# ---------------------------------------------------------------------------

TASK_DISPLAY = {
    "All":                        None,
    "Regulatory Interpretation":  "regulatory_interpretation",
    "Numerical Reasoning":        "numerical_reasoning",
    "Contradiction Detection":    "contradiction_detection",
    "Temporal Reasoning":         "temporal_reasoning",
}

DIFFICULTY_DISPLAY = {
    "All":    None,
    "Easy":   "easy",
    "Medium": "medium",
    "Hard":   "hard",
}


def filter_items(task_label: str, difficulty_label: str) -> list:
    """
    Return dataset items matching the selected task and difficulty filters.

    Parameters
    ----------
    task_label : str
        Display label from the task dropdown (e.g. "Regulatory Interpretation").
    difficulty_label : str
        Display label from the difficulty dropdown (e.g. "Easy").

    Returns
    -------
    list
        Filtered list of dataset records.
    """
    data = load_dataset_once()
    task_key  = TASK_DISPLAY.get(task_label)
    diff_key  = DIFFICULTY_DISPLAY.get(difficulty_label)

    filtered = [
        item for item in data
        if (task_key  is None or item["task_type"]  == task_key)
        and (diff_key is None or item["difficulty"] == diff_key)
    ]
    return filtered if filtered else data  # fallback to all if no match


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

_current_item: dict = {}


def load_random_question(task_label: str, difficulty_label: str):
    """
    Select a random item from the filtered dataset and populate the UI fields.

    Parameters
    ----------
    task_label : str
        Selected task type display label.
    difficulty_label : str
        Selected difficulty display label.

    Returns
    -------
    tuple
        Values to populate: (context, question, meta_label, answer_box)
    """
    global _current_item

    candidates = filter_items(task_label, difficulty_label)
    item       = random.choice(candidates)
    _current_item = item

    task_pretty = item["task_type"].replace("_", " ").title()
    diff_pretty = item["difficulty"].capitalize()
    meta        = f"🏷️  Task: {task_pretty} · Difficulty: {diff_pretty} · Source: {item['source']}"

    return (
        item["context"],    # context_box
        item["question"],   # question_box
        meta,               # meta_label
        "",                 # clear the answer box on each new question
        "",                 # clear the source_doc box
    )


def reveal_answer():
    """
    Reveal the reference answer and source document for the current item.

    Returns
    -------
    tuple
        (reference_answer_text, source_document_text)
    """
    if not _current_item:
        return "Load a question first.", ""
    return (
        _current_item.get("reference_answer", "N/A"),
        f"📄 {_current_item.get('source_document', 'N/A')}",
    )


# ---------------------------------------------------------------------------
# Results table (for About tab)
# ---------------------------------------------------------------------------

RESULTS_TABLE_MD = """
| Model | REG | NUM | CON | TMP | Overall |
|---|---|---|---|---|---|
| Claude 3 Haiku | 92.5% | **93.8%** | 86.7% | **91.4%** | **91.3%** |
| Gemini 2.5 Flash | **96.2%** | 84.4% | 83.3% | 82.4% | 87.9% |
| LLaMA-3.3-70B | 77.4% | 84.4% | **90.0%** | 77.1% | 81.3% |
| LLaMA-3-8B | 77.4% | 62.5% | 86.7% | 74.3% | 75.3% |
| Mistral-7B | 69.8% | 68.8% | 80.0% | 74.3% | 72.7% |

REG = Regulatory Interpretation · NUM = Numerical Reasoning ·
CON = Contradiction Detection · TMP = Temporal Reasoning
"""

ABOUT_MD = f"""
## IndiaFinBench

**The first publicly available evaluation benchmark for LLM performance on
Indian financial regulatory text.**

IndiaFinBench contains **150 expert-annotated question-answer pairs** drawn from
192 documents sourced from the Securities and Exchange Board of India (SEBI) and
the Reserve Bank of India (RBI), spanning documents from 1992 to 2026.

### Four Task Types

| Task | Items | Description |
|---|---|---|
| Regulatory Interpretation | 53 | Extract rules, thresholds, applicability scopes |
| Numerical Reasoning | 32 | Arithmetic over regulatory figures |
| Contradiction Detection | 30 | Identify contradictions between two passages |
| Temporal Reasoning | 35 | Chronological ordering of regulatory events |

### Evaluation Results (Zero-Shot)

{RESULTS_TABLE_MD}

### Key Findings

- **Best model:** Claude 3 Haiku (91.3% overall)
- **Most discriminative task:** Numerical Reasoning (~31 point spread)
- **Dominant frontier-model failure:** Temporal Reasoning (54% of Claude's errors)
- **Dominant small-model failure:** Domain Knowledge (41% of Mistral's errors)

### Links

- 📄 [Paper (arXiv — coming soon)](https://arxiv.org/)
- 💾 [GitHub Repository](https://github.com/Rajveer-code/IndiaFinBench)
- 🤗 [HuggingFace Dataset](https://huggingface.co/datasets/Rajveer-code/IndiaFinBench)

### Citation

```bibtex
@article{{pall2025indiafinbench,
  title={{IndiaFinBench: An Evaluation Benchmark for Large Language Model
         Performance on Indian Financial Regulatory Text}},
  author={{Pall, Rajveer Singh}},
  journal={{arXiv preprint}},
  year={{2025}}
}}
```

### License

Dataset: CC BY 4.0 · Source documents: Public domain (Government of India)
"""

# ---------------------------------------------------------------------------
# UI — gr.Blocks layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="IndiaFinBench — LLM Benchmark Demo",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    .header-box { text-align: center; margin-bottom: 1em; }
    .answer-box textarea { font-weight: bold; background: #f0f7ff; }
    """,
) as demo:

    # ---- Header ----
    with gr.Column(elem_classes="header-box"):
        gr.Markdown(
            "# 🏛️ IndiaFinBench — LLM Benchmark Demo\n"
            "### Evaluating LLM performance on Indian financial regulatory text"
        )

    with gr.Tabs():

        # ================================================================
        # TAB 1 — Interactive Question Browser
        # ================================================================
        with gr.Tab("🔍 Browse Questions"):

            gr.Markdown(
                "Select a **task type** and **difficulty level**, then click "
                "**Load Random Question** to sample a benchmark item. "
                "Try to answer before clicking **Reveal Reference Answer**."
            )

            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=list(TASK_DISPLAY.keys()),
                    value="All",
                    label="Task Type",
                    interactive=True,
                )
                difficulty_dropdown = gr.Dropdown(
                    choices=list(DIFFICULTY_DISPLAY.keys()),
                    value="All",
                    label="Difficulty",
                    interactive=True,
                )

            load_btn = gr.Button("⚡ Load Random Question", variant="primary")

            meta_label = gr.Markdown(
                value="*Load a question to see its metadata.*",
                label="",
            )

            context_box = gr.Textbox(
                label="📜 Context Passage",
                lines=8,
                interactive=False,
                placeholder="Context passage will appear here…",
            )

            question_box = gr.Textbox(
                label="❓ Question",
                lines=3,
                interactive=False,
                placeholder="Question will appear here…",
            )

            reveal_btn = gr.Button("👁️ Reveal Reference Answer", variant="secondary")

            answer_box = gr.Textbox(
                label="✅ Reference Answer",
                lines=2,
                interactive=False,
                placeholder="Click 'Reveal Reference Answer' to show the gold answer…",
                elem_classes="answer-box",
            )

            source_doc_box = gr.Textbox(
                label="📄 Source Document",
                lines=1,
                interactive=False,
                placeholder="Source document will appear here…",
            )

            # ---- Wiring ----
            load_btn.click(
                fn=load_random_question,
                inputs=[task_dropdown, difficulty_dropdown],
                outputs=[context_box, question_box, meta_label, answer_box, source_doc_box],
            )

            reveal_btn.click(
                fn=reveal_answer,
                inputs=[],
                outputs=[answer_box, source_doc_box],
            )

        # ================================================================
        # TAB 2 — About
        # ================================================================
        with gr.Tab("ℹ️ About This Benchmark"):
            gr.Markdown(ABOUT_MD)

    gr.Markdown(
        "---\n*IndiaFinBench is released under CC BY 4.0. "
        "Source documents are public domain (Government of India publications).*"
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
