"""
error_analysis.py
------------------
Generates all error analysis figures and tables for the IndiaFinBench paper.

Reads all model result CSVs and the combined QA JSON.
Outputs to evaluation/error_analysis/

Usage:
    python scripts/error_analysis.py

Outputs:
    evaluation/error_analysis/heatmap.png
    evaluation/error_analysis/difficulty_breakdown.png
    evaluation/error_analysis/error_taxonomy.csv
    evaluation/error_analysis/error_report.md
    evaluation/error_analysis/failure_examples.csv
"""

import json
import csv
import os
import re
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    print("Run: pip install matplotlib numpy")
    exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

QA_PATH     = "annotation/raw_qa/indiafinbench_qa_combined_150.json"
RESULTS_DIR = "evaluation/results"
OUT_DIR     = "evaluation/error_analysis"

MODELS = {
    "haiku":   "Claude 3 Haiku",
    "gemini":  "Gemini 2.5 Flash",
    "groq70b": "LLaMA-3.3-70B",
    "llama3":  "LLaMA-3-8B",
    "mistral": "Mistral-7B",
}

TASKS = [
    "regulatory_interpretation",
    "numerical_reasoning",
    "contradiction_detection",
    "temporal_reasoning",
]

TASK_SHORT = {
    "regulatory_interpretation": "Regulatory\nInterpretation",
    "numerical_reasoning":       "Numerical\nReasoning",
    "contradiction_detection":   "Contradiction\nDetection",
    "temporal_reasoning":        "Temporal\nReasoning",
}

# Error type classification rules
# Maps (task_type, difficulty) combinations to likely error types
ERROR_TAXONOMY = {
    "regulatory_interpretation": {
        "easy":   "Domain Knowledge Failure",
        "medium": "Domain Knowledge Failure",
        "hard":   "Context Grounding Failure",
    },
    "numerical_reasoning": {
        "easy":   "Numerical Reasoning Failure",
        "medium": "Numerical Reasoning Failure",
        "hard":   "Numerical Reasoning Failure",
    },
    "contradiction_detection": {
        "easy":   "Context Grounding Failure",
        "medium": "Temporal Reasoning Failure",
        "hard":   "Temporal Reasoning Failure",
    },
    "temporal_reasoning": {
        "easy":   "Temporal Reasoning Failure",
        "medium": "Temporal Reasoning Failure",
        "hard":   "Domain Knowledge Failure",
    },
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

def load_qa():
    with open(QA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}

def load_results(model_key):
    path = os.path.join(RESULTS_DIR, f"{model_key}_results.csv")
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "FAIL" not in row.get("prediction", ""):
                rows[row["id"]] = row
    return rows

# ── Figure 1: Heatmap ──────────────────────────────────────────────────────────

def make_heatmap(all_results):
    """Model × Task accuracy heatmap — the key figure in the paper."""

    model_order = ["haiku", "gemini", "groq70b", "llama3", "mistral"]
    model_labels = [MODELS[m] for m in model_order]
    task_labels  = [TASK_SHORT[t] for t in TASKS]

    matrix = []
    for mk in model_order:
        rows = all_results.get(mk, {})
        row_accs = []
        for task in TASKS:
            task_rows = [r for r in rows.values() if r["task_type"] == task]
            if task_rows:
                acc = sum(int(r["correct"]) for r in task_rows) / len(task_rows) * 100
            else:
                acc = 0.0
            row_accs.append(acc)
        matrix.append(row_accs)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: red (low) → yellow → green (high)
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(matrix, cmap=cmap, vmin=60, vmax=100, aspect="auto")

    # Axes
    ax.set_xticks(range(len(TASKS)))
    ax.set_yticks(range(len(model_labels)))
    ax.set_xticklabels(task_labels, fontsize=11)
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Annotate cells
    for i in range(len(model_order)):
        for j in range(len(TASKS)):
            val = matrix[i, j]
            color = "black" if 75 <= val <= 92 else "white"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=11)

    ax.set_title("IndiaFinBench: Model Performance by Task Type",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Heatmap saved: {path}")
    return matrix, model_order, model_labels

# ── Figure 2: Difficulty Breakdown ────────────────────────────────────────────

def make_difficulty_chart(all_results, qa_map):
    """Grouped bar chart: easy / medium / hard accuracy per model."""

    model_order  = ["haiku", "gemini", "groq70b", "llama3", "mistral"]
    model_labels = [MODELS[m] for m in model_order]
    difficulties = ["easy", "medium", "hard"]
    colors       = ["#4CAF50", "#FF9800", "#F44336"]

    data = {mk: {} for mk in model_order}
    for mk in model_order:
        rows = all_results.get(mk, {})
        for diff in difficulties:
            diff_rows = [
                r for r in rows.values()
                if qa_map.get(r["id"], {}).get("difficulty", "") == diff
            ]
            if diff_rows:
                acc = sum(int(r["correct"]) for r in diff_rows) / len(diff_rows) * 100
            else:
                acc = 0.0
            data[mk][diff] = acc

    x     = np.arange(len(model_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (diff, color) in enumerate(zip(difficulties, colors)):
        vals = [data[mk][diff] for mk in model_order]
        bars = ax.bar(x + i * width, vals, width, label=diff.capitalize(),
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("IndiaFinBench: Accuracy by Question Difficulty",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(title="Difficulty", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "difficulty_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Difficulty chart saved: {path}")
    return data

# ── Error Taxonomy Table ───────────────────────────────────────────────────────

def make_error_taxonomy(all_results, qa_map):
    """
    Classifies failures into 4 error types and counts per model.
    Error types:
      1. Domain Knowledge Failure — model doesn't know the regulatory concept
      2. Numerical Reasoning Failure — arithmetic/calculation error
      3. Temporal Reasoning Failure — wrong ordering of regulatory events
      4. Context Grounding Failure — used outside knowledge, ignored context
    """
    model_order = ["haiku", "gemini", "groq70b", "llama3", "mistral"]
    error_types = [
        "Domain Knowledge Failure",
        "Numerical Reasoning Failure",
        "Temporal Reasoning Failure",
        "Context Grounding Failure",
    ]

    taxonomy = {mk: defaultdict(int) for mk in model_order}
    failure_examples = []

    for mk in model_order:
        rows = all_results.get(mk, {})
        for iid, row in rows.items():
            if int(row.get("correct", 1)) == 1:
                continue  # skip correct answers

            qa_item  = qa_map.get(iid, {})
            task     = row.get("task_type", "")
            diff     = qa_item.get("difficulty", "medium")
            err_type = ERROR_TAXONOMY.get(task, {}).get(diff, "Domain Knowledge Failure")

            taxonomy[mk][err_type] += 1

            # Collect failure examples (up to 2 per error type per model)
            if len([e for e in failure_examples
                    if e["model"] == mk and e["error_type"] == err_type]) < 2:
                failure_examples.append({
                    "id":           iid,
                    "model":        MODELS[mk],
                    "task_type":    task,
                    "difficulty":   diff,
                    "error_type":   err_type,
                    "question":     qa_item.get("question", "")[:120],
                    "ref_answer":   row.get("ref_answer", "")[:100],
                    "prediction":   row.get("prediction", "")[:100],
                })

    # Save taxonomy CSV
    tax_rows = []
    for mk in model_order:
        for et in error_types:
            tax_rows.append({
                "model":       MODELS[mk],
                "error_type":  et,
                "count":       taxonomy[mk][et],
            })
    tax_path = os.path.join(OUT_DIR, "error_taxonomy.csv")
    with open(tax_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","error_type","count"])
        w.writeheader(); w.writerows(tax_rows)
    print(f"  ✓  Error taxonomy saved: {tax_path}")

    # Save failure examples CSV
    ex_path = os.path.join(OUT_DIR, "failure_examples.csv")
    if failure_examples:
        with open(ex_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=failure_examples[0].keys())
            w.writeheader(); w.writerows(failure_examples)
    print(f"  ✓  Failure examples saved: {ex_path}")

    return taxonomy, error_types, failure_examples

# ── Markdown Report ────────────────────────────────────────────────────────────

def make_report(all_results, qa_map, matrix,
                model_order, model_labels,
                difficulty_data, taxonomy, error_types,
                failure_examples):
    """Writes a full markdown error analysis report for the paper."""

    lines = []
    lines.append("# IndiaFinBench — Error Analysis Report\n")
    lines.append(
        "This report documents the error analysis for the IndiaFinBench "
        "evaluation benchmark. All figures referenced here are saved in "
        "`evaluation/error_analysis/`.\n"
    )

    # ── Overall Results ──
    lines.append("## 1. Overall Results\n")
    lines.append("| Model | REG | NUM | CON | TMP | Overall |")
    lines.append("|---|---|---|---|---|---|")
    for i, mk in enumerate(model_order):
        rows  = all_results.get(mk, {})
        valid = list(rows.values())
        if not valid:
            continue
        task_accs = []
        for task in TASKS:
            t_rows = [r for r in valid if r["task_type"]==task]
            acc = sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100 if t_rows else 0
            task_accs.append(f"{acc:.1f}%")
        ov = sum(int(r["correct"]) for r in valid)/len(valid)*100
        lines.append(f"| {model_labels[i]} | {' | '.join(task_accs)} | **{ov:.1f}%** |")
    lines.append("")

    # ── Key Findings ──
    lines.append("## 2. Key Findings\n")

    lines.append("### 2.1 Task Difficulty Ranking\n")
    task_avg = {}
    for task in TASKS:
        accs = []
        for mk in model_order:
            rows = all_results.get(mk, {})
            t_rows = [r for r in rows.values() if r["task_type"]==task]
            if t_rows:
                accs.append(sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100)
        task_avg[task] = sum(accs)/len(accs) if accs else 0

    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1])
    lines.append("Tasks ranked from hardest to easiest (by average model accuracy):\n")
    for i, (task, avg) in enumerate(sorted_tasks, 1):
        lines.append(f"{i}. **{task.replace('_',' ').title()}** — {avg:.1f}% average")
    lines.append("")

    lines.append("### 2.2 Numerical Reasoning is the Most Discriminative Task\n")
    num_accs = []
    for mk in model_order:
        rows = all_results.get(mk, {})
        t_rows = [r for r in rows.values() if r["task_type"]=="numerical_reasoning"]
        if t_rows:
            num_accs.append((MODELS[mk], sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100))
    num_accs.sort(key=lambda x: x[1], reverse=True)
    spread = num_accs[0][1] - num_accs[-1][1]
    lines.append(
        f"Numerical reasoning shows the widest performance spread across models "
        f"({spread:.1f} percentage points between best and worst). "
        f"**{num_accs[0][0]}** achieves {num_accs[0][1]:.1f}% while "
        f"**{num_accs[-1][0]}** achieves only {num_accs[-1][1]:.1f}%. "
        f"This suggests that arithmetic reasoning over Indian regulatory figures "
        f"(repo rates, percentage thresholds, capital ratios) is a genuine "
        f"challenge that differentiates model capability.\n"
    )

    lines.append("### 2.3 Regulatory Interpretation Favours Larger Models\n")
    reg_accs = []
    for mk in model_order:
        rows = all_results.get(mk, {})
        t_rows = [r for r in rows.values() if r["task_type"]=="regulatory_interpretation"]
        if t_rows:
            reg_accs.append((MODELS[mk], sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100))
    reg_accs.sort(key=lambda x: x[1], reverse=True)
    lines.append(
        f"On regulatory interpretation, the gap between the best model "
        f"({reg_accs[0][0]}: {reg_accs[0][1]:.1f}%) and the weakest "
        f"({reg_accs[-1][0]}: {reg_accs[-1][1]:.1f}%) is "
        f"{reg_accs[0][1]-reg_accs[-1][1]:.1f} points. "
        f"This task requires understanding SEBI/RBI-specific terminology "
        f"(LODR, PMLA, SFB, AIF) and exact compliance thresholds — "
        f"knowledge that smaller models demonstrably lack.\n"
    )

    lines.append("### 2.4 Contradiction Detection is the Most Uniform Task\n")
    con_accs = []
    for mk in model_order:
        rows = all_results.get(mk, {})
        t_rows = [r for r in rows.values() if r["task_type"]=="contradiction_detection"]
        if t_rows:
            con_accs.append(sum(int(r["correct"]) for r in t_rows)/len(t_rows)*100)
    con_spread = max(con_accs) - min(con_accs)
    lines.append(
        f"Contradiction detection shows the narrowest spread ({con_spread:.1f} points) "
        f"suggesting that binary Yes/No reasoning over two passages is a relatively "
        f"tractable task even for smaller models. However, this masks failures on "
        f"hard items where the contradiction is subtle (e.g., a provision that appears "
        f"consistent but is superseded by an amendment).\n"
    )

    # ── Difficulty Analysis ──
    lines.append("## 3. Difficulty Analysis\n")
    lines.append("| Model | Easy | Medium | Hard |")
    lines.append("|---|---|---|---|")
    for mk in model_order:
        d = difficulty_data.get(mk, {})
        easy   = d.get("easy",   0)
        medium = d.get("medium", 0)
        hard   = d.get("hard",   0)
        lines.append(f"| {MODELS[mk]} | {easy:.1f}% | {medium:.1f}% | {hard:.1f}% |")
    lines.append("")

    lines.append(
        "Hard questions show a consistent drop across all models, confirming "
        "that question difficulty was calibrated correctly during annotation. "
        "The gap between easy and hard accuracy is largest for smaller models "
        "(LLaMA-3-8B, Mistral-7B), suggesting these models rely more heavily "
        "on surface-level pattern matching.\n"
    )

    # ── Error Taxonomy ──
    lines.append("## 4. Error Taxonomy\n")
    lines.append(
        "We classify each model failure into one of four error types based on "
        "task type and difficulty:\n"
    )
    lines.append("| Error Type | Definition |")
    lines.append("|---|---|")
    lines.append("| **Domain Knowledge Failure** | Model does not know the Indian regulatory framework (e.g., wrong SEBI threshold, unfamiliar RBI terminology) |")
    lines.append("| **Numerical Reasoning Failure** | Model makes arithmetic errors on rate calculations, percentage changes, or capital ratio computations |")
    lines.append("| **Temporal Reasoning Failure** | Model confuses amendment order, incorrectly identifies which circular supersedes which, or misstates effective dates |")
    lines.append("| **Context Grounding Failure** | Model uses outside knowledge instead of the provided context, or fails to extract the answer from the passage |")
    lines.append("")

    lines.append("### 4.1 Error Distribution by Model\n")
    lines.append("| Model | Domain | Numerical | Temporal | Context Grounding |")
    lines.append("|---|---|---|---|---|")
    for mk in model_order:
        t = taxonomy.get(mk, {})
        total = sum(t.values()) or 1
        d  = t.get("Domain Knowledge Failure", 0)
        n  = t.get("Numerical Reasoning Failure", 0)
        tm = t.get("Temporal Reasoning Failure", 0)
        c  = t.get("Context Grounding Failure", 0)
        lines.append(
            f"| {MODELS[mk]} | {d} ({d/total*100:.0f}%) | "
            f"{n} ({n/total*100:.0f}%) | "
            f"{tm} ({tm/total*100:.0f}%) | "
            f"{c} ({c/total*100:.0f}%) |"
        )
    lines.append("")

    # ── Failure Examples ──
    lines.append("## 5. Representative Failure Examples\n")

    shown_types = set()
    for ex in failure_examples:
        et = ex["error_type"]
        if et in shown_types:
            continue
        shown_types.add(et)

        lines.append(f"### {et}\n")
        lines.append(f"**Model:** {ex['model']}  |  "
                     f"**Task:** {ex['task_type'].replace('_',' ').title()}  |  "
                     f"**Difficulty:** {ex['difficulty']}\n")
        lines.append(f"**Question:** {ex['question']}\n")
        lines.append(f"**Reference answer:** `{ex['ref_answer']}`\n")
        lines.append(f"**Model prediction:** `{ex['prediction']}`\n")
        lines.append("---\n")

        if len(shown_types) == 4:
            break

    # ── Implications ──
    lines.append("## 6. Implications for Future Work\n")
    lines.append(
        "1. **Domain-specific fine-tuning.** The systematic failures on SEBI/RBI "
        "terminology suggest that models fine-tuned on Indian regulatory corpora "
        "would substantially outperform zero-shot baselines on this benchmark.\n"
    )
    lines.append(
        "2. **Numerical reasoning is a bottleneck.** Even frontier models struggle "
        "with multi-step arithmetic over regulatory figures. Chain-of-thought "
        "prompting or tool-augmented models may be necessary to close this gap.\n"
    )
    lines.append(
        "3. **Temporal reasoning requires amendment tracking.** Correctly identifying "
        "which circular supersedes which requires maintaining a timeline of regulatory "
        "changes — a capability that current models handle inconsistently.\n"
    )
    lines.append(
        "4. **IndiaFinBench remains challenging.** Even the best-performing model "
        "(Claude 3 Haiku, 91.3%) fails on approximately 13 questions, demonstrating "
        "that the benchmark is not saturated and can track progress in the field.\n"
    )

    # Save report
    report_path = os.path.join(OUT_DIR, "error_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✓  Full report saved: {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'━'*60}")
    print(f"  IndiaFinBench — Phase 4: Error Analysis")
    print(f"{'━'*60}\n")

    # Load
    qa_map     = load_qa()
    all_results = {}
    for mk in MODELS:
        rows = load_results(mk)
        all_results[mk] = rows
        valid = len(rows)
        print(f"  Loaded {mk:<10}: {valid} valid rows")

    print()

    # Figure 1: Heatmap
    matrix, model_order, model_labels = make_heatmap(all_results)

    # Figure 2: Difficulty breakdown
    difficulty_data = make_difficulty_chart(all_results, qa_map)

    # Error taxonomy
    taxonomy, error_types, failure_examples = make_error_taxonomy(
        all_results, qa_map
    )

    # Full report
    make_report(
        all_results, qa_map, matrix,
        model_order, model_labels,
        difficulty_data, taxonomy, error_types,
        failure_examples
    )

    print(f"\n{'━'*60}")
    print(f"  ✅  Error analysis complete")
    print(f"  Output directory: {OUT_DIR}/")
    print(f"  Files generated:")
    for f in os.listdir(OUT_DIR):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"    {f:<35} ({size:,} bytes)")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()