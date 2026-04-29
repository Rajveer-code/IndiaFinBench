"""
append_batch.py
---------------
Merges a batch of new questions into an existing QA JSON file.
Validates IDs, checks for duplicates, and writes the combined output.

Usage:
    python scripts/append_batch.py \\
        --existing annotation/raw_qa/indiafinbench_qa_combined_150.json \\
        --batch    batch1_new_questions.json \\
        --out      annotation/raw_qa/indiafinbench_qa_combined_193.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Merge a question batch into the main dataset")
    parser.add_argument("--existing", required=True, help="Existing combined JSON path (relative to project root)")
    parser.add_argument("--batch",    required=True, help="New batch JSON path (relative to project root)")
    parser.add_argument("--out",      required=True, help="Output JSON path (relative to project root)")
    args = parser.parse_args()

    existing_path = ROOT / args.existing
    batch_path    = ROOT / args.batch
    out_path      = ROOT / args.out

    # Load existing
    with open(existing_path, encoding="utf-8") as f:
        existing = json.load(f)
    print(f"Existing dataset : {len(existing)} items")

    # Load batch
    with open(batch_path, encoding="utf-8") as f:
        batch = json.load(f)
    print(f"Batch to merge   : {len(batch)} items")

    # Validate: check for ID conflicts
    existing_ids = {item["id"] for item in existing}
    batch_ids    = [item["id"] for item in batch]

    # Check for duplicates within the batch
    batch_id_counter = Counter(batch_ids)
    internal_dups = {k: v for k, v in batch_id_counter.items() if v > 1}
    if internal_dups:
        print(f"ERROR: Duplicate IDs within batch: {internal_dups}", file=sys.stderr)
        sys.exit(1)

    # Check for conflicts with existing
    conflicts = [id_ for id_ in batch_ids if id_ in existing_ids]
    if conflicts:
        print(f"ERROR: {len(conflicts)} IDs already exist in dataset:", file=sys.stderr)
        for c in conflicts[:10]:
            print(f"  {c}", file=sys.stderr)
        sys.exit(1)

    # Validate required fields
    # CON items use document_a/context_a instead of document/context
    REQUIRED_BASE = {"id", "task_type", "source", "question", "answer", "difficulty"}
    REQUIRED_SINGLE = {"document", "context"}
    REQUIRED_CON = {"document_a", "context_a"}
    for i, item in enumerate(batch):
        missing_base = REQUIRED_BASE - set(item.keys())
        has_single = REQUIRED_SINGLE <= set(item.keys())
        has_con = REQUIRED_CON <= set(item.keys())
        if missing_base or (not has_single and not has_con):
            missing = missing_base | (REQUIRED_SINGLE if not has_single and not has_con else set())
            print(f"ERROR: Item {i} (id={item.get('id','?')}) missing fields: {missing}", file=sys.stderr)
            sys.exit(1)

    # Merge
    combined = existing + batch

    # Summary
    tt   = Counter(item["task_type"] for item in combined)
    diff = Counter(item["difficulty"] for item in combined)
    ids  = {}
    for item in combined:
        p = item["id"].split("_")[0]
        try:
            n = int(item["id"].split("_")[1])
        except (IndexError, ValueError):
            continue
        ids[p] = max(ids.get(p, 0), n)

    print(f"Combined total   : {len(combined)} items")
    print(f"Task types       : {dict(tt)}")
    print(f"Difficulty       : {dict(diff)}")
    print(f"Max IDs          : {ids}")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Output saved to  : {out_path}")


if __name__ == "__main__":
    main()
