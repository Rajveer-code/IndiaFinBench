"""
parse_pdfs.py
-------------
Converts all downloaded PDFs (SEBI + RBI) into clean text files.

Uses pdfplumber (better than PyPDF2 for multi-column layouts,
tables, and scanned-adjacent PDFs common in Indian regulatory docs).

For each PDF in data/raw/sebi/ and data/raw/rbi/, outputs a
corresponding .txt file in data/parsed/sebi/ or data/parsed/rbi/

Also generates data/parse_report.csv showing:
  - filename, pages, word_count, status (success/failed/empty)

Usage:
  python scripts/parse_pdfs.py

Outputs:
  data/parsed/sebi/*.txt
  data/parsed/rbi/*.txt
  data/parse_report.csv
"""

import os
import re
import csv
import pdfplumber
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

SOURCES = [
    {"raw_dir": "data/raw/sebi",  "parsed_dir": "data/parsed/sebi"},
    {"raw_dir": "data/raw/rbi",   "parsed_dir": "data/parsed/rbi"},
]

REPORT_PATH   = "data/parse_report.csv"
MIN_WORDS     = 50    # Discard parsed text with fewer words (likely scanned/image PDF)

# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Cleans extracted PDF text:
      - Removes excessive whitespace and blank lines
      - Removes page headers/footers (common patterns in SEBI/RBI docs)
      - Normalises unicode dashes and quotes
      - Strips non-printable characters
    """
    if not raw:
        return ""

    # Remove non-printable characters (but keep Hindi/Devanagari unicode)
    text = "".join(
        ch for ch in raw
        if ch.isprintable() or ch in "\n\t"
    )

    # Normalise dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Remove lines that are just page numbers (e.g. "1", "Page 2 of 10")
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip pure page number lines
        if re.fullmatch(r"(Page\s+)?\d+(\s+of\s+\d+)?", stripped, re.IGNORECASE):
            continue
        # Skip very short lines that are likely headers/footers
        if len(stripped) < 3:
            continue
        cleaned_lines.append(stripped)

    # Collapse multiple blank lines into one
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── PDF Parsing ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> dict:
    """
    Parses a single PDF and returns:
      {
        text:       full cleaned text string,
        pages:      number of pages,
        word_count: word count after cleaning,
        status:     'success' | 'empty' | 'failed',
        error:      error message if failed
      }
    """
    result = {
        "text":       "",
        "pages":      0,
        "word_count": 0,
        "status":     "failed",
        "error":      "",
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["pages"] = len(pdf.pages)
            all_text_parts = []

            for page in pdf.pages:
                # Extract text — pdfplumber preserves layout better than PyPDF2
                page_text = page.extract_text(
                    x_tolerance=3,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13,
                )
                if page_text:
                    all_text_parts.append(page_text)

            raw_text = "\n\n".join(all_text_parts)
            cleaned  = clean_text(raw_text)
            words    = len(cleaned.split())

            result["text"]       = cleaned
            result["word_count"] = words
            result["status"]     = "success" if words >= MIN_WORDS else "empty"

    except Exception as e:
        result["error"]  = str(e)[:200]
        result["status"] = "failed"

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    report_rows = []
    total_success = 0
    total_failed  = 0
    total_empty   = 0

    print(f"\n{'━'*55}")
    print(f"  PDF Parser — IndiaFinBench")
    print(f"{'━'*55}\n")

    for source in SOURCES:
        raw_dir    = source["raw_dir"]
        parsed_dir = source["parsed_dir"]
        source_name = os.path.basename(raw_dir).upper()  # 'SEBI' or 'RBI'

        os.makedirs(parsed_dir, exist_ok=True)

        pdf_files = [
            f for f in os.listdir(raw_dir)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            print(f"  ⚠  No PDFs found in {raw_dir}")
            continue

        print(f"  [{source_name}] Parsing {len(pdf_files)} PDFs ...")

        for filename in tqdm(pdf_files, desc=f"  {source_name}"):
            pdf_path = os.path.join(raw_dir, filename)
            txt_name = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(parsed_dir, txt_name)

            # Skip if already parsed
            if os.path.exists(txt_path) and os.path.getsize(txt_path) > 10:
                report_rows.append({
                    "source":    source_name,
                    "filename":  filename,
                    "txt_file":  txt_name,
                    "pages":     "—",
                    "word_count": "—",
                    "status":    "skipped",
                    "error":     "",
                    "parsed_at": "—",
                })
                total_success += 1
                continue

            result = parse_pdf(pdf_path)

            if result["status"] == "success":
                # Write the clean text file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                total_success += 1

            elif result["status"] == "empty":
                # Write it anyway but flag it — useful to inspect manually
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                print(f"\n    ⚠  Low word count ({result['word_count']} words): {filename}")
                total_empty += 1

            else:
                print(f"\n    ✗  Parse failed: {filename} — {result['error']}")
                total_failed += 1

            report_rows.append({
                "source":     source_name,
                "filename":   filename,
                "txt_file":   txt_name,
                "pages":      result["pages"],
                "word_count": result["word_count"],
                "status":     result["status"],
                "error":      result["error"],
                "parsed_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    # Write report
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"\n{'━'*55}")
    print(f"  ✅  Parsing complete")
    print(f"  Success : {total_success}")
    print(f"  Empty   : {total_empty}  (low word count — check manually)")
    print(f"  Failed  : {total_failed}")
    print(f"  Report  : {REPORT_PATH}")
    print(f"{'━'*55}\n")

    if total_empty > 0:
        print("  ℹ  'Empty' usually means a scanned PDF with no text layer.")
        print("     These can be fixed with OCR (we'll handle this if needed).\n")


if __name__ == "__main__":
    main()