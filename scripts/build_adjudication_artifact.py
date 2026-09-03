"""
build_adjudication_artifact.py
------------------------------
Builds the shareable "Second Opinion" web page (annotation/independent_adjudication/
artifact/template.html + the frozen blind_sheet.csv) into a single self-contained HTML
file ready to publish as a Claude Artifact.

Fixes applied to the source data before embedding:
- blind_sheet.csv has one row (NUM_033) with mojibake from an earlier CSV write step
  (UTF-8 em-dash bytes misread as cp1252 and re-encoded) -- repaired via the standard
  cp1252-encode/utf-8-decode round trip, display-layer only, frozen source untouched.
- All non-ASCII text is emitted as \\uXXXX JSON escapes (json.dump default, ensure_ascii
  not overridden) rather than raw UTF-8 bytes, so the page can't be corrupted by a
  serving environment that doesn't declare charset=utf-8 (caught during testing: raw
  UTF-8 arrows/em-dashes in the page's own JS literals rendered as mojibake under a
  bare `python -m http.server`, which sends no charset header -- \\u escapes sidestep
  the problem entirely since JS decodes them from pure ASCII source).

Note: blind_sheet.csv has 8 repeated item_ids (same benchmark question, different
models' disputed predictions) -- the page keys answers by array INDEX, not item_id,
and the exported CSV includes a 1-based `row` column so results can be merged back
unambiguously. Do not key anything in the page by item_id alone.

Usage: python scripts/build_adjudication_artifact.py
Output: build/adjudication_final.html (gitignored; publish this file as the Artifact)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / "annotation/independent_adjudication/artifact/template.html"
BLIND_SHEET = ROOT / "annotation/independent_adjudication/blind_sheet.csv"
OUT = ROOT / "build/adjudication_final.html"


def fix_mojibake(s: str) -> str:
    if "â€" in s or "€" in s:
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def main():
    with open(BLIND_SHEET, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    items = [{
        "id": r["item_id"], "task": r["task_type"],
        "q": fix_mojibake(r["question"]), "ref": fix_mojibake(r["reference_answer"]),
        "pred": fix_mojibake(r["model_prediction"]),
    } for r in rows]

    items_json = json.dumps(items)  # ensure_ascii=True default -- see module docstring
    html = TEMPLATE.read_text(encoding="utf-8")
    final = html.replace("__ITEMS_JSON__", items_json)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(final, encoding="utf-8")
    data = OUT.read_bytes()
    non_ascii = sum(1 for b in data if b > 127)
    print(f"Wrote {OUT} ({len(data)} bytes, {len(items)} items, {non_ascii} non-ASCII bytes)")
    if non_ascii:
        print("WARNING: non-ASCII bytes present -- check the template for raw Unicode literals.")


if __name__ == "__main__":
    main()
