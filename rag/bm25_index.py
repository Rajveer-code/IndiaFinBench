"""
rag/bm25_index.py
-----------------
BM25Okapi index for lexical retrieval over chunk texts.

Tokenisation strategy:
  Lowercase + strip punctuation EXCEPT hyphens and parentheses.
  Rationale: Indian regulatory text contains tokens like "91-day", "4(2)(b)",
  "Section-51A" whose internal punctuation carries semantic meaning.
  Standard whitespace tokenisation after these targeted strips preserves them.

BM25 parameters:
  k1=1.5  — term frequency saturation (standard Okapi default)
  b=0.75  — document length normalisation (standard Okapi default)
  These are not tuned; the corpus is too small to warrant grid search.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.models import ChunkRecord


# Strip punctuation that is NOT part of regulatory term identifiers
_STRIP_RE = re.compile(r"[^\w\s\-()]")


class BM25Index:
    def __init__(self) -> None:
        self._bm25:   BM25Okapi | None   = None
        self._chunks: list[ChunkRecord]  = []

    # ── Tokenisation ──────────────────────────────────────────────────────────

    @staticmethod
    def tokenise(text: str) -> list[str]:
        text = text.lower()
        text = _STRIP_RE.sub(" ", text)
        return text.split()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[ChunkRecord]) -> None:
        self._chunks = list(chunks)
        tokenised    = [self.tokenise(c.text) for c in chunks]
        self._bm25   = BM25Okapi(tokenised, k1=1.5, b=0.75)

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int) -> list[tuple[ChunkRecord, float]]:
        """Return up to k (chunk, bm25_score) pairs, descending by score."""
        if self._bm25 is None:
            raise RuntimeError("BM25Index not built. Call build() or load() first.")
        tokens = self.tokenise(query)
        scores = self._bm25.get_scores(tokens)
        top_k  = min(k, len(self._chunks))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in ranked]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "bm25.pkl", "wb") as fh:
            pickle.dump((self._bm25, self._chunks), fh)

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        obj = cls()
        with open(Path(index_dir) / "bm25.pkl", "rb") as fh:
            obj._bm25, obj._chunks = pickle.load(fh)
        return obj

    @property
    def size(self) -> int:
        return len(self._chunks)
