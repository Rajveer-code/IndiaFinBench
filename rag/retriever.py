"""
rag/retriever.py
----------------
Hybrid retriever combining dense (FAISS) and lexical (BM25) search via
Reciprocal Rank Fusion (Cormack et al., 2009).

RRF formula:
    score(d) = Σ_{r ∈ {dense, bm25}} 1 / (k_RRF + rank_r(d))
where k_RRF=60 (empirically optimal constant from the original paper).
Chunks absent from a list receive rank = candidates + 1.

Two additional constraints:
  - Source diversity: at most max_per_source chunks from the same regulatory
    source ("rbi" or "sebi") in the final top-k, to support cross-document
    synthesis queries.
  - Score floor: dense-only and bm25-only modes supported for ablation
    (pass mode="dense" | "bm25" | "hybrid").
"""

from rag.bm25_index import BM25Index
from rag.embeddings import BGEEmbedder
from rag.index import FAISSIndex
from rag.models import ChunkRecord, RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        faiss_index:    FAISSIndex,
        bm25_index:     BM25Index,
        embedder:       BGEEmbedder,
        top_k:          int = 5,
        candidates:     int = 20,
        rrf_k:          int = 60,
        max_per_source: int = 3,
    ) -> None:
        self._faiss         = faiss_index
        self._bm25          = bm25_index
        self._embedder      = embedder
        self.top_k          = top_k
        self.candidates     = candidates
        self.rrf_k          = rrf_k
        self.max_per_source = max_per_source

    def retrieve(
        self, query: str, mode: str = "hybrid"
    ) -> list[RetrievalResult]:
        """
        mode: "hybrid" | "dense" | "bm25"
          Used in Phase 3 ablation to isolate individual retriever contributions.
        """
        absent_rank = self.candidates + 1

        # ── Dense retrieval ───────────────────────────────────────────────────
        dense_hits: list[tuple[ChunkRecord, float]] = []
        if mode in ("hybrid", "dense"):
            qemb       = self._embedder.encode_query(query)
            dense_hits = self._faiss.search(qemb, self.candidates)

        # ── BM25 retrieval ────────────────────────────────────────────────────
        bm25_hits: list[tuple[ChunkRecord, float]] = []
        if mode in ("hybrid", "bm25"):
            bm25_hits = self._bm25.search(query, self.candidates)

        # ── Build rank & score lookup maps ────────────────────────────────────
        dense_rank  = {c.chunk_id: r for r, (c, _) in enumerate(dense_hits, 1)}
        bm25_rank   = {c.chunk_id: r for r, (c, _) in enumerate(bm25_hits, 1)}
        dense_score = {c.chunk_id: s for c, s in dense_hits}
        bm25_score  = {c.chunk_id: s for c, s in bm25_hits}
        chunk_map   = {c.chunk_id: c for c, _ in dense_hits + bm25_hits}

        # ── RRF scoring ───────────────────────────────────────────────────────
        rrf_scores: dict[str, float] = {}
        for cid in chunk_map:
            rd = dense_rank.get(cid, absent_rank)
            rb = bm25_rank.get(cid, absent_rank)
            if mode == "dense":
                rrf_scores[cid] = 1.0 / (self.rrf_k + rd)
            elif mode == "bm25":
                rrf_scores[cid] = 1.0 / (self.rrf_k + rb)
            else:
                rrf_scores[cid] = 1.0 / (self.rrf_k + rd) + 1.0 / (self.rrf_k + rb)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Source diversity cap + top-k selection ────────────────────────────
        results:      list[RetrievalResult] = []
        source_count: dict[str, int]        = {}

        for cid, rrf in ranked:
            if len(results) >= self.top_k:
                break
            chunk  = chunk_map[cid]
            n_from = source_count.get(chunk.source, 0)
            if n_from >= self.max_per_source:
                continue
            source_count[chunk.source] = n_from + 1
            results.append(RetrievalResult(
                chunk       = chunk,
                dense_score = dense_score.get(cid, 0.0),
                bm25_score  = bm25_score.get(cid, 0.0),
                rrf_score   = rrf,
                dense_rank  = dense_rank.get(cid, absent_rank),
                bm25_rank   = bm25_rank.get(cid, absent_rank),
            ))

        return results
