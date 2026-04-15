"""
Experiment 6: Text Feature Difficulty Regression
Identify which linguistic properties of items predict model failure.
"""
import sys, warnings, re
warnings.filterwarnings('ignore')
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import (
    load_dataset, load_all_results, OUTPUT_DIR, FIGURES_DIR, _correctness_col
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")
OUTPUT = OUTPUT_DIR / "feature_regression"
OUTPUT.mkdir(parents=True, exist_ok=True)


def extract_features(row):
    """Extract linguistic features from a QA item."""
    context = str(row.get('context', '') or '')
    question = str(row.get('question', '') or '')
    combined = context + ' ' + question

    ctx_tokens = context.split()

    dates = re.findall(
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b\d{4}\b',
        combined
    )
    numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', combined)
    currencies = re.findall(r'₹|crore|lakh|cr\.?|Rs\.?', combined, re.IGNORECASE)
    amendment_words = re.findall(r'\bsupersed|\bamend|\bmodif|\breplace|\bhereby\b', combined, re.IGNORECASE)
    sebi_rbi = re.findall(r'\bSEBI\b|\bRBI\b|\bCircular\b|\bNotification\b', combined)
    conditions = re.findall(
        r'\bif\b|\bwhere\b|\bprovided that\b|\bsubject to\b|\bnotwithstanding\b',
        combined, re.IGNORECASE
    )

    try:
        flesch = textstat.flesch_reading_ease(context[:1000]) if len(context) > 50 else 50.0
        fk_grade = textstat.flesch_kincaid_grade(context[:1000]) if len(context) > 50 else 10.0
    except Exception:
        flesch, fk_grade = 50.0, 10.0

    return {
        'ctx_len': len(ctx_tokens),
        'q_len': len(question.split()),
        'num_dates': len(set(dates)),
        'num_numbers': len(numbers),
        'num_currencies': len(currencies),
        'num_amendments': len(amendment_words),
        'num_sebi_rbi_refs': len(sebi_rbi),
        'num_conditions': len(conditions),
        'flesch_ease': flesch,
        'fk_grade': fk_grade,
        'sentence_count': combined.count('.') + combined.count(';'),
    }


def main():
    print("Loading data...")
    dataset = load_dataset()
    all_results = load_all_results()

    print("Extracting linguistic features (this may take 2-3 minutes)...")
    features = dataset.apply(extract_features, axis=1)
    features_df = pd.DataFrame(list(features))
    feature_cols = features_df.columns.tolist()
    print(f"Extracted {len(feature_cols)} features for {len(dataset)} items")

    corr_cols = []
    for model_name, res_df in all_results.items():
        corr_c = _correctness_col(res_df)
        col_name = f'corr_{model_name.replace(" ", "_")}'
        features_df[col_name] = res_df[corr_c].values[:len(features_df)]
        corr_cols.append(col_name)

    features_df['ensemble_accuracy'] = features_df[corr_cols].mean(axis=1)
    features_df['is_hard'] = (features_df['ensemble_accuracy'] < 0.5).astype(int)

    if 'task_type' in dataset.columns:
        features_df['task_type'] = dataset['task_type'].values
    if 'difficulty' in dataset.columns:
        features_df['human_difficulty'] = dataset['difficulty'].values

    features_df.to_csv(OUTPUT / "item_features.csv", index=False)

    # ── Logistic Regression ───────────────────────────────────────────────
    X = features_df[feature_cols].fillna(0)
    y = features_df['is_hard']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
    print(f"\nLogistic Regression AUC={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X_scaled, y)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0],
        'abs_coef': np.abs(model.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    coef_df.to_csv(OUTPUT / "feature_importance.csv", index=False)
    print("\nTop features predicting item difficulty:")
    print(coef_df.head(10).to_string(index=False))

    # ── Per-task correlations ─────────────────────────────────────────────
    task_correlations = []
    if 'task_type' in features_df.columns:
        for task in features_df['task_type'].unique():
            task_mask = features_df['task_type'] == task
            task_data = features_df[task_mask]
            if len(task_data) < 10:
                continue
            for feat in feature_cols:
                r, p = stats.spearmanr(
                    task_data[feat].fillna(0),
                    1 - task_data['ensemble_accuracy']
                )
                task_correlations.append({'task': task, 'feature': feat, 'r': round(r, 3), 'p': round(p, 4)})

    if task_correlations:
        pd.DataFrame(task_correlations).to_csv(OUTPUT / "task_feature_correlations.csv", index=False)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    top10 = coef_df.head(10)
    colors = ['#F44336' if c > 0 else '#4CAF50' for c in top10['coefficient']]
    ax.barh(top10['feature'], top10['coefficient'], color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Logistic Regression Coefficient', fontsize=11)
    ax.set_title('Features Predicting Item Difficulty\n(Red=harder, Green=easier)',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(features_df['num_dates'], features_df['ensemble_accuracy'],
                alpha=0.5, s=30, c=features_df['ensemble_accuracy'], cmap='RdYlGn')
    ax2.set_xlabel('Number of Dates in Item', fontsize=11)
    ax2.set_ylabel('Ensemble Accuracy', fontsize=11)
    ax2.set_title('Date Density vs. Item Difficulty\n(All 406 items)', fontsize=11, fontweight='bold')
    z = np.polyfit(features_df['num_dates'].fillna(0), features_df['ensemble_accuracy'].fillna(0.5), 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(0, features_df['num_dates'].max(), 100)
    ax2.plot(x_range, p_line(x_range), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "exp6_feature_regression.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    print("=== EXPERIMENT 6 COMPLETE ===")
    print(f"Top difficulty predictor: {coef_df.iloc[0]['feature']} (coef={coef_df.iloc[0]['coefficient']:.3f})")


if __name__ == "__main__":
    main()
