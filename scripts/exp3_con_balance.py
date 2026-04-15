"""
Experiment 3: CON Class Balance and Baseline Analysis
Checks if contradiction detection results are inflated by class imbalance.

Dataset field: answer (contains Yes/No for CON items)
"""
import sys, json
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, OUTPUT_DIR, FIGURES_DIR,
    _correctness_col, _task_col
)
import matplotlib.pyplot as plt

OUTPUT = OUTPUT_DIR / "con_balance"
OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset...")
    dataset = load_dataset()

    con_mask = dataset['task_type'].str.contains('contradiction', case=False, na=False)
    con_items = dataset[con_mask].copy()
    print(f"Found {len(con_items)} CON items")

    if len(con_items) == 0:
        print("Available task types:", dataset['task_type'].unique())
        return

    # Use 'answer' field (verified dataset field name)
    ref_col = 'answer'
    if ref_col not in con_items.columns:
        # Fallback: find any answer-like column
        candidates = [c for c in con_items.columns if 'answer' in c.lower()]
        ref_col = candidates[0] if candidates else con_items.columns[0]

    answers = con_items[ref_col].astype(str).str.lower().str.strip()
    yes_count = int(answers.str.startswith('yes').sum())
    no_count = int(answers.str.startswith('no').sum())
    total = len(con_items)

    print(f"\nCON class distribution:")
    print(f"  Yes (contradiction exists): {yes_count} ({yes_count/total:.1%})")
    print(f"  No (no contradiction):      {no_count}  ({no_count/total:.1%})")

    majority_class = 'Yes' if yes_count > no_count else 'No'
    majority_baseline = max(yes_count, no_count) / total
    random_baseline = 0.50

    print(f"\nBaselines:")
    print(f"  Majority class ({majority_class}): {majority_baseline:.1%}")
    print(f"  Random (50/50): {random_baseline:.1%}")

    all_results = load_all_results()
    model_con_acc = {}
    for model_name, res_df in all_results.items():
        task_c = _task_col(res_df)
        corr_c = _correctness_col(res_df)
        con_res = res_df[res_df[task_c].str.contains('contradiction', case=False, na=False)]
        if len(con_res) > 0:
            model_con_acc[model_name] = float(con_res[corr_c].mean())

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.pie(
        [yes_count, no_count],
        labels=[f'Yes\n({yes_count/total:.1%})', f'No\n({no_count/total:.1%})'],
        colors=['#F44336', '#4CAF50'], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12}
    )
    ax.set_title('CON Task: Class Distribution\n(Yes = contradiction exists)',
                 fontsize=12, fontweight='bold')

    ax2 = axes[1]
    sorted_models = sorted(model_con_acc.items(), key=lambda x: x[1])
    model_names_sorted = [m[0] for m in sorted_models]
    model_accs = [m[1] for m in sorted_models]
    colors = ['#90CAF9' if acc < majority_baseline + 0.05 else '#2196F3'
              for acc in model_accs]
    ax2.barh(model_names_sorted, model_accs, color=colors, alpha=0.85)
    ax2.axvline(x=majority_baseline, color='red', linestyle='--', linewidth=2,
                label=f'Majority baseline ({majority_baseline:.1%})')
    ax2.axvline(x=random_baseline, color='orange', linestyle=':', linewidth=2,
                label=f'Random baseline (50.0%)')
    ax2.set_xlabel('CON Task Accuracy', fontsize=12)
    ax2.set_title('Model CON Accuracy vs. Baselines', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 1.1)
    ax2.grid(axis='x', alpha=0.3)
    for i, (name, acc) in enumerate(sorted_models):
        ax2.text(acc + 0.01, i, f'{acc:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp3_con_balance.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    summary = {
        'yes_count': yes_count,
        'no_count': no_count,
        'total': total,
        'majority_class': majority_class,
        'majority_baseline': round(majority_baseline, 4),
        'random_baseline': round(random_baseline, 4),
        'model_accuracies': {k: round(v, 4) for k, v in model_con_acc.items()}
    }
    with open(OUTPUT / "con_balance_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    above = sum(1 for v in model_con_acc.values() if v > majority_baseline + 0.05)
    print("\n=== EXPERIMENT 3 COMPLETE ===")
    print(f"Key finding: Majority baseline = {majority_baseline:.1%}. Models above this: {above}")


if __name__ == "__main__":
    main()
