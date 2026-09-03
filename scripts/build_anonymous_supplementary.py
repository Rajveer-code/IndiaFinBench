"""
build_anonymous_supplementary.py
------------------------------
Plan v3 Phase 9.7: package the essential reproducibility material into an
anonymized ZIP for TMLR's supplementary-material upload (up to 100MB, must
itself be anonymized -- the anon.4open.science mirror going stale once
already is exactly why this shouldn't be the only artifact).

Includes only what's needed to reproduce every table/figure: the dataset,
evaluation harness, per-item verdicts for all three regimes, and the analysis
scripts. Deliberately excludes anything identity-bearing (README.md, HF/GitHub
upload scripts, docs/, demo/, _archive/, old paper drafts, .git) rather than
trying to redact them in place -- exclusion is simpler and safer than redaction.

Usage: python scripts/build_anonymous_supplementary.py
Output: build/tmlr_supplementary.zip (gitignored, not committed)
"""
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
STAGE = ROOT / "build" / "tmlr_supplementary_stage"
ZIP_PATH = ROOT / "build" / "tmlr_supplementary.zip"

# Identity strings that must not appear anywhere in the staged copy.
IDENTITY_STRINGS = [
    "Rajveer", "rajveerpall", "0009-0001-6762-6134", "Rajveer-code",
]

# Top-level items to include, as (source relative path, dest relative path).
INCLUDE_DIRS = ["annotation", "evaluation", "scripts"]
INCLUDE_FILES = ["requirements.txt"]

# Files/patterns to exclude even within an included directory -- identity-
# bearing utility scripts that aren't part of the reproducibility path.
EXCLUDE_NAMES = {
    "upload_to_huggingface.py", "deploy_space.sh", "generate_paper_docx.py",
    "patch_docx.py", "patch_docx2.py",
    "build_anonymous_supplementary.py",  # this script itself -- a build tool, not part of reproduction
    "build_adjudication_artifact.py",  # builds the named-recipient share page, not reproduction
}
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_DIR_NAMES = {
    "__pycache__", "_novel_methods_BACKUP_11models_2026-08-31",
    "artifact",  # annotation/independent_adjudication/artifact/ names the author by design
}

SUPPLEMENTARY_README = """\
# IndiaFinBench -- Anonymized Supplementary Material

This package reproduces every table and figure in the submitted manuscript from raw data.
Author identity has been removed per TMLR's anonymity requirement; see the manuscript's own
double-blind submission for the full paper text.

## Contents

- `annotation/` -- the 406-item dataset (raw_qa/), inter-annotator-agreement study (iaa/),
  judge-disagreement adjudication sheets (judge_audit/, independent_adjudication/).
- `evaluation/` -- per-model prediction CSVs (results/, results_matched/, results_judged_phi4/)
  and every derived analysis JSON (regime_three_way.json, matched_budget_comparison.json,
  item_discrimination_exact.json, bootstrap significance results, etc.).
- `scripts/` -- the full evaluation harness and every analysis script that produces a number,
  table, or figure appearing in the manuscript.
- `requirements.txt` -- pinned Python dependencies.

## Regenerating a headline result

```bash
pip install -r requirements.txt
python scripts/regime_table.py              # reproduces the three-regime table (Table 1)
python scripts/matched_budget_comparison.py # reproduces the matched-budget table (Table 2)
python scripts/item_discrimination_exact.py # reproduces the discriminative-coverage table (Table 3)
```

Each script reads only from `evaluation/` and `annotation/` in this package and writes its
output as JSON/LaTeX -- no network access or external credentials required for any of the
three commands above (they operate on already-collected data).

## What's not included

Model-calling code paths that require third-party API credentials (Groq, OpenRouter, Gemini,
Vertex AI) are present in `scripts/` for transparency but are not needed to reproduce any
reported number -- all model outputs are already saved under `evaluation/`.
"""


def is_excluded(path: Path) -> bool:
    if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return True
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def copy_filtered(src: Path, dst: Path):
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        if is_excluded(item):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    for d in INCLUDE_DIRS:
        src = ROOT / d
        if src.exists():
            copy_filtered(src, STAGE / d)
    for f in INCLUDE_FILES:
        src = ROOT / f
        if src.exists():
            shutil.copy2(src, STAGE / f)

    (STAGE / "README.md").write_text(SUPPLEMENTARY_README, encoding="utf-8")

    # Identity scan before zipping -- fail loudly rather than ship a leak.
    hits = []
    for item in STAGE.rglob("*"):
        if item.is_dir():
            continue
        try:
            text = item.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for s in IDENTITY_STRINGS:
            if s in text:
                hits.append((str(item.relative_to(STAGE)), s))

    if hits:
        print("IDENTITY LEAK FOUND -- not zipping. Fix these files first:")
        for path, s in hits:
            print(f"  {path}: contains {s!r}")
        raise SystemExit(1)

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in STAGE.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(STAGE))

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    n_files = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    print(f"Zero identity-string hits across {n_files} staged files.")
    print(f"Wrote {ZIP_PATH} ({size_mb:.1f} MB, {n_files} files)")
    if size_mb > 100:
        print("WARNING: exceeds TMLR's 100MB supplementary-material limit.")


if __name__ == "__main__":
    main()
