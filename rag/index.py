"""
rag/index.py
------------
FAISS IndexFlatIP wrapper for exact cosine-similarity retrieval.

Why IndexFlatIP?
  At M ≈ 7,000 chunks × 768 dims the index is ~21.5 MB and a single query
  costs ~1.35 ms on CPU (5.4M FP32 multiplications). Approximate indices
  (IVF, HNSW) introduce recall loss without meaningful latency benefit at
  this scale.

Embeddings MUST be L2-normalised before add() so that inner product = cosine.
The BGEEmbedder guarantees this; do not pass raw embeddings.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.models import ChunkRecord


class FAISSIndex:
    def __init__(self, dim: int) -> None:
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[ChunkRecord] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, chunks: list[ChunkRecord]) -> None:
        if embeddings.shape != (len(chunks), self.dim):
            raise ValueError(
                f"Embedding shape {embeddings.shape} does not match "
                f"({len(chunks)}, {self.dim})"
            )
        self.index.add(embeddings)
        self._chunks = list(chunks)

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, k: int
    ) -> list[tuple[ChunkRecord, float]]:
        """
        Return up to k (chunk, cosine_score) pairs, descending by score.
        query_embedding must be shape (1, dim) and L2-normalised.
        """
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        results: list[tuple[ChunkRecord, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "wb") as fh:
            pickle.dump(self._chunks, fh)

    @classmethod
    def load(cls, index_dir: Path, dim: int) -> "FAISSIndex":
        index_dir = Path(index_dir)
        obj       = cls(dim)
        obj.index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "rb") as fh:
            obj._chunks = pickle.load(fh)
        return obj

    @property
    def size(self) -> int:
        return self.index.ntotal
