"""
rag/preprocessing.py
--------------------
Lightweight text normalisation for PDF-extracted Indian regulatory documents.
All operations are pure string transforms — no model inference.

Pipeline (applied in order):
  1. Unicode NFKC normalisation  — resolve ligatures, non-breaking spaces
  2. Header/footer line removal  — heuristic pattern match on short lines
  3. Trailing whitespace strip   — per-line
  4. Blank-line collapse         — 3+ consecutive blank lines → 2
  5. Leading/trailing strip      — final document trim
"""

import re
import unicodedata


# Lines matching these patterns and under 80 chars are dropped.
# Ordered from most to least specific to minimise false positives.
_HEADER_FOOTER_PATTERNS = re.compile(
    r"""
    ^\s*(?:
        (?:reserve\s+bank\s+of\s+india)             |
        (?:securities\s+and\s+exchange\s+board)      |
        (?:rbi\s*[/|–-])                             |
        (?:sebi\s*[/|–-])                            |
        (?:page\s+\d+\s*(?:of\s+\d+)?)              |
        (?:\d+\s*$)                                  |   # bare page numbers
        (?:www\.\S+)                                 |
        (?:https?://\S+)                             |
        (?:©\s*.+)                                   |
        (?:circular\s+no\.?\s*[a-z0-9/_\-\.]+$)
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


class TextPreprocessor:
    def process(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = self._strip_headers_footers(text)
        text = _TRAILING_SPACE.sub("", text)
        text = _MULTI_BLANK.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def _strip_headers_footers(text: str) -> str:
        lines = text.splitlines()
        cleaned: list[str] = []
        for line in lines:
            # Only apply heuristic to short lines to avoid false positives
            if len(line) < 80 and _HEADER_FOOTER_PATTERNS.match(line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)
