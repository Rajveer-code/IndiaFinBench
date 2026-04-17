"""
Generate a 90-item annotation validation sheet for multi-annotator IAA study.

Stratified sample: 30 REG + 15 NUM + 15 CON + 30 TMP
Format: annotators see context + question + reference_answer
They fill ONE column: is_correct__YES_or_NO  (and optional notes if NO)

Output files:
  annotation/multi_annotator/annotation_sheet_90items.csv  ← send to annotators
  annotation/multi_annotator/annotation_reference.csv      ← internal reference
"""
import sys
sys.path.append(str(__import__('pathlib').Path(__file__).parent.parent))
from scripts.novel_methods_utils import load_dataset
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
OUT_DIR = REPO_ROOT / "annotation/multi_annotator"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZES = {
    'regulatory_interpretation': 30,
    'numerical_reasoning':       15,
    'contradiction_detection':   15,
    'temporal_reasoning':        30,
}


def build_context_display(row) -> str:
    """Format context for display in the sheet."""
    task = str(row.get('task_type', '')).lower()
    if 'contradiction' in task:
        ctx_a = str(row.get('context_a', row.get('context', '')))[:500]
        ctx_b = str(row.get('context_b', ''))[:300]
        return f"PASSAGE A:\n{ctx_a}\n\nPASSAGE B:\n{ctx_b}"
    return str(row.get('context', ''))[:700]


def main():
    dataset = load_dataset()
    print(f"Dataset loaded: {len(dataset)} items")

    sampled_parts = []
    for task, n in SAMPLE_SIZES.items():
        task_df = dataset[
            dataset['task_type'].str.contains(task, case=False, na=False)
        ].copy()

        if len(task_df) == 0:
            print(f"  WARNING: no items found for task={task}")
            continue

        # Stratify by difficulty if available
        if 'difficulty' in task_df.columns and task_df['difficulty'].nunique() > 1:
            diffs = task_df['difficulty'].unique()
            per_diff = max(1, n // len(diffs))
            parts = []
            for diff in diffs:
                d_df = task_df[task_df['difficulty'] == diff]
                k = min(per_diff, len(d_df))
                parts.append(d_df.sample(k, random_state=42))
            task_sample = pd.concat(parts).head(n)
        else:
            task_sample = task_df.sample(min(n, len(task_df)), random_state=42)

        sampled_parts.append(task_sample)
        print(f"  {task}: {len(task_sample)} items sampled")

    sheet_raw = pd.concat(sampled_parts).drop_duplicates('id').reset_index(drop=True)
    print(f"Total items in sheet: {len(sheet_raw)}")

    # Build the annotator-facing sheet
    annotator_sheet = pd.DataFrame({
        'row_number': range(1, len(sheet_raw) + 1),
        'id':         sheet_raw['id'],
        'task_type':  sheet_raw['task_type'],
        'difficulty': sheet_raw.get('difficulty', 'unknown'),
        'context':    sheet_raw.apply(build_context_display, axis=1),
        'question':   sheet_raw['question'],
        'reference_answer': sheet_raw['answer'],
        'is_correct__YES_or_NO': '',   # ← annotator fills this
        'notes_if_NO':          '',    # ← optional
    })

    sheet_path = OUT_DIR / "annotation_sheet_90items.csv"
    annotator_sheet.to_csv(sheet_path, index=False)
    print(f"\nAnnotator sheet saved → {sheet_path}")

    # Internal reference (item IDs + gold answers, for computing Fleiss κ later)
    ref = sheet_raw[['id', 'task_type', 'difficulty', 'answer']].copy()
    ref.to_csv(OUT_DIR / "annotation_reference.csv", index=False)
    print(f"Reference answers saved → {OUT_DIR}/annotation_reference.csv")

    print("\n" + "="*60)
    print("INSTRUCTIONS TO SEND TO YOUR ANNOTATORS:")
    print("="*60)
    print("""
Open annotation_sheet_90items.csv in Excel or Google Sheets.
For each row:
  1. Read the CONTEXT and QUESTION
  2. Read the REFERENCE_ANSWER
  3. In column 'is_correct__YES_or_NO': type YES or NO
     (YES = the reference answer is correct given only the context)
     (NO  = the reference answer is wrong or incomplete)
  4. If NO, optionally add a brief note in 'notes_if_NO'

Do NOT look at other rows before answering.
Takes about 45 minutes.
""")

    print(f"Task distribution:")
    print(annotator_sheet['task_type'].value_counts().to_string())


if __name__ == "__main__":
    main()
