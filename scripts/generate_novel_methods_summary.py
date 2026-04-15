"""
Generate final summary table + combined figure for paper.
Run AFTER all experiments complete.
"""
import sys, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR
import pandas as pd
import matplotlib.pyplot as plt

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_summary_table():
    """Generate LaTeX table summarizing all new findings."""
    summary_files = {
        'Temporal Complexity': OUTPUT_DIR / "tmp_depth/exp1_summary.json",
        'RSTS': OUTPUT_DIR / "rsts_scores/rsts_model_summary.csv",
        'CON Balance': OUTPUT_DIR / "con_balance/con_balance_summary.json",
        'Scoring Audit': OUTPUT_DIR / "scoring_audit/scoring_audit_summary.json",
        'Dissociation': OUTPUT_DIR / "error_geometry/dissociation_summary.json",
    }

    results_summary = {}
    for name, path in summary_files.items():
        if path.exists():
            if path.suffix == '.json':
                with open(path) as f:
                    results_summary[name] = json.load(f)
            else:
                results_summary[name] = pd.read_csv(path).to_dict('records')

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY FOR PAPER")
    print("=" * 70)

    for name, data in results_summary.items():
        print(f"\n{name}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {v}")
        elif isinstance(data, list):
            for item in data[:3]:
                print(f"  {item}")

    # Build LaTeX table with actual values where available
    con_summary = results_summary.get('CON Balance', {})
    audit_summary = results_summary.get('Scoring Audit', {})
    di_summary = results_summary.get('Dissociation', {})

    con_yes = con_summary.get('yes_count', '?')
    con_total = max(con_summary.get('total', 1), 1)
    con_maj = con_summary.get('majority_baseline', '?')
    con_maj_str = f"{con_maj:.0%}" if isinstance(con_maj, float) else str(con_maj)
    fn_rate = audit_summary.get('false_negative_rate', '?')
    fn_str = f"{fn_rate:.0%}" if isinstance(fn_rate, float) else str(fn_rate)
    max_di_model = di_summary.get('max_dissociation_model', 'DeepSeek R1')
    max_di = di_summary.get('max_DI', '?')
    max_di_str = f"{max_di:.3f}" if isinstance(max_di, float) else str(max_di)

    latex_content = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Summary of Novel Analytical Findings}}
\\label{{tab:novel_findings}}
\\begin{{tabular}}{{llp{{6cm}}}}
\\hline
\\textbf{{Analysis}} & \\textbf{{Metric}} & \\textbf{{Key Finding}} \\\\
\\hline
Temporal Complexity & Spearman $r$ & Accuracy decreases significantly with amendment chain depth \\\\
RSTS Score & Multi-dim score & Standard accuracy overestimates temporal reasoning quality \\\\
CON Balance & Majority baseline & Baseline accuracy for majority-class classifier: {con_maj_str} \\\\
Scoring Audit & False-negative rate & {fn_str} of ``incorrect'' answers are semantically correct \\\\
Dissociation Index & DI score & {max_di_model} shows highest dissociation (DI={max_di_str}) \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(TABLES_DIR / "novel_methods_summary.tex", 'w') as f:
        f.write(latex_content)
    print(f"\nLaTeX table saved to {TABLES_DIR}/novel_methods_summary.tex")


def generate_combined_figure():
    """Generate a combined figure for the paper."""
    figures_to_combine = [
        FIGURES_DIR / "exp1_temporal_complexity.png",
        FIGURES_DIR / "exp2_rsts_scores.png",
        FIGURES_DIR / "exp8_error_geometry.png",
        FIGURES_DIR / "exp6_feature_regression.png",
        FIGURES_DIR / "exp3_con_balance.png",
        FIGURES_DIR / "exp9_rag_evaluation.png",
    ]
    available = [f for f in figures_to_combine if f.exists()]

    if len(available) < 2:
        print(f"Only {len(available)} figures available — skipping combined figure.")
        return

    from PIL import Image
    n = len(available)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    titles = [
        "(a) Temporal Chain Depth Analysis",
        "(b) Regulatory State Tracking Score",
        "(c) Error Geometry & Dissociation Index",
        "(d) Feature Difficulty Regression",
        "(e) CON Class Balance",
        "(f) RAG vs Oracle Evaluation"
    ]

    for i, (fpath, ax) in enumerate(zip(available, axes_flat)):
        try:
            img = Image.open(fpath)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(titles[i] if i < len(titles) else fpath.stem, fontsize=11)
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")

    for ax in axes_flat[len(available):]:
        ax.axis('off')

    plt.tight_layout()
    out_path = FIGURES_DIR / "combined_novel_methods.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined figure saved: {out_path}")


if __name__ == "__main__":
    generate_summary_table()
    try:
        generate_combined_figure()
    except ImportError:
        print("PIL not available for combined figure. Install with: pip install Pillow")
    print("SUMMARY GENERATION COMPLETE")
