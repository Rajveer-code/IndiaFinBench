"""
demo/app.py
-----------
Purpose:  Full IndiaFinBench leaderboard on HuggingFace Spaces.
          Tab 1 — Live sortable leaderboard (pre-populated with 5 baselines).
          Tab 2 — Submit: enter a HF model ID, get auto-scored on all 150 items.
          Tab 3 — About: paper abstract, dataset stats, citation.
Inputs:   demo/data/questions.json, demo/data/baselines.json, demo/leaderboard.db
Outputs:  Interactive Gradio web app
Usage:
    python demo/app.py
    gradio demo/app.py
"""

import json
import random
import sys
import threading
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure demo/ is on the Python path when running from repo root
_DEMO_DIR = Path(__file__).parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from database.db import get_leaderboard, init_db, save_result
from evaluation.scorer import score_submission
from evaluation.tasks import build_prompt, extract_answer

# ── Globals ────────────────────────────────────────────────────────────────────

QUESTIONS_PATH = _DEMO_DIR / "data/questions.json"

with QUESTIONS_PATH.open(encoding="utf-8") as _f:
    QUESTIONS: list[dict] = json.load(_f)

TASK_FULL = {
    "regulatory_interpretation": "Regulatory Interpretation",
    "numerical_reasoning":       "Numerical Reasoning",
    "contradiction_detection":   "Contradiction Detection",
    "temporal_reasoning":        "Temporal Reasoning",
}

# Initialise DB (creates tables + inserts baselines if needed)
init_db()

# ── Leaderboard helpers ────────────────────────────────────────────────────────

def refresh_leaderboard() -> pd.DataFrame:
    """Reload the leaderboard DataFrame from SQLite.

    Returns:
        Sorted leaderboard DataFrame.
    """
    return get_leaderboard()


def build_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Build an overall-accuracy bar chart from the leaderboard DataFrame.

    Args:
        df: Leaderboard DataFrame returned by get_leaderboard().

    Returns:
        Matplotlib Figure.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center")
        return fig

    models  = df["Model"].tolist()
    overall = df["Overall (%)"].tolist()

    palette = plt.cm.Blues(np.linspace(0.4, 0.8, len(models)))
    fig, ax = plt.subplots(figsize=(max(6, len(models)), 4))
    bars = ax.bar(models, overall, color=palette, edgecolor="white")

    for bar, val in zip(bars, overall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("IndiaFinBench Leaderboard — Overall Accuracy", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# ── Submit helpers ─────────────────────────────────────────────────────────────

_eval_lock = threading.Lock()  # Prevent concurrent evaluations on Spaces


def validate_hf_id(hf_id: str) -> tuple[bool, str]:
    """Basic validation of a HuggingFace model ID.

    Args:
        hf_id: Model ID string (e.g. "mistralai/Mistral-7B-Instruct-v0.3").

    Returns:
        (is_valid, error_message) tuple.
    """
    hf_id = hf_id.strip()
    if not hf_id:
        return False, "Please enter a HuggingFace model ID."
    if len(hf_id) > 200:
        return False, "Model ID is too long."
    return True, ""


def run_evaluation(
    hf_id: str,
    display_name: str,
    params_str: str,
    smoke_test: bool,
) -> tuple[str, pd.DataFrame, plt.Figure]:
    """Run evaluation of a HuggingFace model against IndiaFinBench.

    Args:
        hf_id:        HuggingFace model ID.
        display_name: Display label for the leaderboard.
        params_str:   Parameter count string (e.g. "7B").
        smoke_test:   If True, evaluate only first 10 items.

    Returns:
        (status_message, leaderboard_df, bar_chart_figure)
    """
    hf_id = hf_id.strip()
    valid, err = validate_hf_id(hf_id)
    if not valid:
        return err, refresh_leaderboard(), build_bar_chart(refresh_leaderboard())

    label = display_name.strip() if display_name.strip() else hf_id.split("/")[-1]

    if not _eval_lock.acquire(blocking=False):
        return (
            "Another evaluation is already running. Please wait a moment and try again.",
            refresh_leaderboard(),
            build_bar_chart(refresh_leaderboard()),
        )

    try:
        from evaluation.evaluator import IndiaFinBenchEvaluator

        n_items    = 10 if smoke_test else len(QUESTIONS)
        eval_items = QUESTIONS[:n_items]

        evaluator = IndiaFinBenchEvaluator(hf_id)
        preds     = evaluator.run(eval_items)

        result    = score_submission(preds, eval_items)
        overall   = result["overall"]
        per_task  = result["per_task"]

        save_result(
            hf_id=hf_id,
            label=label,
            overall=overall,
            per_task=per_task,
            params=params_str or "Unknown",
            n_items=n_items,
            notes="smoke_test" if smoke_test else "",
        )

        status = (
            f"Evaluation complete!\n"
            f"Model: {label}\n"
            f"Overall: {overall*100:.1f}%\n"
            f"REG: {per_task.get('REG',0)*100:.1f}%  "
            f"NUM: {per_task.get('NUM',0)*100:.1f}%  "
            f"CON: {per_task.get('CON',0)*100:.1f}%  "
            f"TMP: {per_task.get('TMP',0)*100:.1f}%\n"
            f"Items evaluated: {n_items}/150"
        )
        df  = refresh_leaderboard()
        return status, df, build_bar_chart(df)

    except Exception as e:
        return (
            f"Evaluation failed: {str(e)[:300]}",
            refresh_leaderboard(),
            build_bar_chart(refresh_leaderboard()),
        )
    finally:
        _eval_lock.release()


# ── Dataset Explorer helpers ───────────────────────────────────────────────────

def get_random_example(task_filter: str, diff_filter: str) -> str:
    """Return a formatted markdown block for a random filtered example.

    Args:
        task_filter: Display label for task type, or "All".
        diff_filter: Difficulty level ("Easy", "Medium", "Hard", "All").

    Returns:
        Markdown-formatted example string.
    """
    pool = list(QUESTIONS)
    if task_filter != "All":
        pool = [q for q in pool
                if TASK_FULL.get(q["task_type"], "") == task_filter]
    if diff_filter != "All":
        pool = [q for q in pool if q["difficulty"] == diff_filter.lower()]

    if not pool:
        return "No examples match the selected filters."

    q    = random.choice(pool)
    task = TASK_FULL.get(q["task_type"], q["task_type"])

    ctx = q.get("context") or (
        f"**Passage A:** {q.get('context_a','')}\n\n"
        f"**Passage B:** {q.get('context_b','')}"
    )

    return (
        f'**Task:** `{task}` &nbsp;&nbsp; **Difficulty:** `{q["difficulty"].upper()}`\n\n'
        f"---\n\n"
        f"**Context:**\n\n> {ctx[:800]}{'...' if len(ctx) > 800 else ''}\n\n"
        f"---\n\n"
        f"**Question:** {q['question']}\n\n"
        f"**Reference Answer:** {q['gold_answer']}"
    )


# ── About content ──────────────────────────────────────────────────────────────

ABOUT_MD = """
## Abstract

**IndiaFinBench** is the first benchmark for evaluating Large Language Models on
Indian financial regulatory text. It comprises **150 expert-annotated QA pairs** from
192 SEBI and RBI documents, covering four task types:

| Task | N | Description |
|---|---|---|
| Regulatory Interpretation (REG) | 53 | Extract rules, thresholds, compliance conditions |
| Numerical Reasoning (NUM) | 32 | Arithmetic over repo rates, capital ratios, thresholds |
| Contradiction Detection (CON) | 30 | Identify contradictions between two regulatory passages |
| Temporal Reasoning (TMP) | 35 | Sequence amendments and effective dates |

## Dataset Statistics

| Statistic | Value |
|---|---|
| Total QA pairs | 150 |
| Difficulty (Easy / Medium / Hard) | 50 / 60 / 40 |
| SEBI source documents | 142 |
| RBI source documents | 89 |
| Inter-annotator agreement | 90.7% |
| Contradiction Cohen's kappa | 0.918 |

## How to Submit

1. Enter your HuggingFace model ID in the **Submit** tab (e.g. `mistralai/Mistral-7B-Instruct-v0.3`)
2. Optionally run a 10-item smoke test first
3. Click **Run Evaluation** — the model is loaded, run zero-shot against all 150 items, and the result is saved to the leaderboard

**Note:** Evaluation requires the model to be publicly accessible on HuggingFace.
Large models (>30B) may time out on Spaces CPU — use GPU-enabled Spaces or run locally.

## Citation

```bibtex
@inproceedings{indiafinbench2026,
  title     = {{IndiaFinBench}: A Benchmark for Evaluating LLMs
               on Indian Financial Regulatory Text},
  author    = {Pall, Rajveer Singh},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026},
}
```

## Links

- **Paper:** arXiv (link TBD after review)
- **GitHub:** Repository link TBD
- **HuggingFace Dataset:** Coming soon
"""


# ── Build Gradio app ───────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Build the full 3-tab Gradio leaderboard app.

    Returns:
        Configured gr.Blocks instance.
    """
    init_df  = refresh_leaderboard()
    init_fig = build_bar_chart(init_df)

    with gr.Blocks(
        title="IndiaFinBench Leaderboard",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "# IndiaFinBench\n"
            "### The first LLM benchmark for Indian financial regulatory text\n"
            "150 QA pairs · SEBI + RBI corpus · 4 task types · Zero-shot evaluation"
        )

        with gr.Tabs():

            # ── Tab 1: Leaderboard ─────────────────────────────────────────────
            with gr.Tab("Leaderboard"):
                gr.Markdown("### Model Rankings (click column headers to sort)")
                lb_table = gr.Dataframe(
                    value=init_df,
                    interactive=False,
                    wrap=False,
                )
                refresh_btn = gr.Button("Refresh Leaderboard", size="sm")
                gr.Markdown("### Overall Accuracy")
                lb_chart = gr.Plot(value=init_fig)

                refresh_btn.click(
                    fn=lambda: (refresh_leaderboard(), build_bar_chart(refresh_leaderboard())),
                    inputs=[],
                    outputs=[lb_table, lb_chart],
                )

            # ── Tab 2: Submit ──────────────────────────────────────────────────
            with gr.Tab("Submit a Model"):
                gr.Markdown(
                    "Enter a public HuggingFace model ID to evaluate it zero-shot "
                    "on all 150 IndiaFinBench questions. Results are added to the leaderboard.\n\n"
                    "> **Tip:** Run a smoke test (10 items) first to check the model loads correctly."
                )

                with gr.Row():
                    hf_id_box = gr.Textbox(
                        label="HuggingFace Model ID",
                        placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.3",
                    )
                    name_box = gr.Textbox(
                        label="Display Name (optional)",
                        placeholder="e.g. Mistral-7B",
                    )

                with gr.Row():
                    params_box = gr.Textbox(
                        label="Parameter Count (optional)",
                        placeholder="e.g. 7B",
                    )
                    smoke_chk = gr.Checkbox(
                        label="Smoke test only (10 items)",
                        value=True,
                    )

                run_btn    = gr.Button("Run Evaluation", variant="primary")
                status_out = gr.Textbox(label="Status", lines=6, interactive=False)
                sub_table  = gr.Dataframe(label="Updated Leaderboard",
                                          interactive=False, wrap=False)
                sub_chart  = gr.Plot(label="Updated Chart")

                run_btn.click(
                    fn=run_evaluation,
                    inputs=[hf_id_box, name_box, params_box, smoke_chk],
                    outputs=[status_out, sub_table, sub_chart],
                )

            # ── Tab 3: About ───────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown(ABOUT_MD)

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
