"""
rag/pipeline.py
---------------
RAGPipeline: the top-level orchestrator that wires ingestion, indexing,
retrieval, and generation into a single callable interface.

Public API (intentionally minimal, mirrors demo/rag/rag.py):
    pipe = RAGPipeline()
    pipe.build_index()   # run once; serialises to rag/index/
    pipe.load_index()    # fast path on subsequent starts
    result = pipe.ask("What are the KYC norms under SEBI?")
    # result: {"answer": str, "sources": list[dict]} | {"error": str}

The demo integration in demo/rag/rag.py is a thin shim over this module.
"""

import logging
from pathlib import Path

from rag.bm25_index import BM25Index
from rag.chunking import RecursiveCharacterSplitter
from rag.config import RAGConfig
from rag.data_loader import DataLoader
from rag.embeddings import BGEEmbedder
from rag.generator import LLMGenerator
from rag.index import FAISSIndex
from rag.models import RetrievalResult
from rag.preprocessing import TextPreprocessor
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None) -> None:
        self.cfg       = config or RAGConfig()
        self._embedder = BGEEmbedder(
            model_name  = self.cfg.embedding_model,
            device      = self.cfg.embedding_device,
            batch_size  = self.cfg.embedding_batch_size,
        )
        self._generator = None

        if getattr(self.cfg, "enable_generation", True):
            self._generator = LLMGenerator(
                backend     = self.cfg.llm_backend,
                max_tokens  = self.cfg.max_tokens,
                temperature = self.cfg.temperature,
        )
        self._retriever: HybridRetriever | None = None

    # ── Index build (offline) ─────────────────────────────────────────────────

    def build_index(self) -> None:
        """
        Full ingestion pipeline: load → preprocess → chunk → embed → index.
        Idempotent: safe to re-run; overwrites rag/index/ on disk.
        Typical runtime on CPU: 2–4 minutes for the 192-document corpus.
        """
        loader      = DataLoader(self.cfg.data_dir)
        preprocessor = TextPreprocessor()
        splitter    = RecursiveCharacterSplitter(
            target_chunk_chars = self.cfg.target_chunk_chars,
            overlap_chars      = self.cfg.overlap_chars,
            min_chunk_chars    = self.cfg.min_chunk_chars,
        )

        docs = loader.load()
        logger.info("Loaded %d documents", len(docs))

        for doc in docs:
            doc.raw_text = preprocessor.process(doc.raw_text)

        all_chunks = []
        for doc in docs:
            all_chunks.extend(splitter.split_document(doc))
        logger.info("Created %d chunks", len(all_chunks))

        texts      = [c.text for c in all_chunks]
        embeddings = self._embedder.encode_corpus(texts)

        faiss_idx = FAISSIndex(self._embedder.dim)
        faiss_idx.build(embeddings, all_chunks)
        faiss_idx.save(self.cfg.index_dir)

        bm25_idx = BM25Index()
        bm25_idx.build(all_chunks)
        bm25_idx.save(self.cfg.index_dir)

        self._wire_retriever(faiss_idx, bm25_idx)
        logger.info(
            "Index built: %d vectors in FAISS, %d chunks in BM25. Saved to %s.",
            faiss_idx.size, bm25_idx.size, self.cfg.index_dir,
        )

    # ── Index load (online startup) ───────────────────────────────────────────

    def load_index(self) -> None:
        faiss_idx = FAISSIndex.load(self.cfg.index_dir, self._embedder.dim)
        bm25_idx  = BM25Index.load(self.cfg.index_dir)
        self._wire_retriever(faiss_idx, bm25_idx)
        logger.info(
            "Index loaded: %d vectors (%s).",
            faiss_idx.size, self.cfg.index_dir,
        )

    def _wire_retriever(self, faiss_idx: FAISSIndex, bm25_idx: BM25Index) -> None:
        self._retriever = HybridRetriever(
            faiss_index    = faiss_idx,
            bm25_index     = bm25_idx,
            embedder       = self._embedder,
            top_k          = self.cfg.top_k,
            candidates     = self.cfg.candidates,
            rrf_k          = self.cfg.rrf_k,
            max_per_source = self.cfg.max_per_source,
        )

    # ── Query (online) ────────────────────────────────────────────────────────

    def ask(self, query: str, mode: str = "hybrid") -> dict:
        """
        Full RAG pipeline: retrieve → generate.

        Args:
            query: Natural language question.
            mode:  "hybrid" | "dense" | "bm25" — retriever mode for ablation.

        Returns:
            {"answer": str, "sources": list[dict]}   on success
            {"error": str}                            on failure
        """
        if not query or not query.strip():
            return {"error": "Empty query."}
        if self._retriever is None:
            return {"error": "Index not loaded. Call build_index() or load_index() first."}

        try:
            results: list[RetrievalResult] = self._retriever.retrieve(
                query.strip(), mode=mode
            )
            if self._generator is None:
                return {"error": "Generation disabled in current configuration."}

            answer = self._generator.generate(query.strip(), results)
            sources = [
                {
                    "chunk_id":    r.chunk.chunk_id,
                    "doc_id":      r.chunk.doc_id,
                    "title":       r.chunk.title,
                    "source":      r.chunk.source,
                    "text":        r.chunk.text,
                    "rrf_score":   round(r.rrf_score,   6),
                    "dense_score": round(r.dense_score, 6),
                    "bm25_score":  round(r.bm25_score,  4),
                }
                for r in results
            ]
            return {"answer": answer, "sources": sources}

        except Exception as exc:  # noqa: BLE001
            logger.exception("RAG pipeline error for query: %r", query)
            return {"error": f"RAG pipeline error: {exc!s}"[:400]}

    # ── Index status ──────────────────────────────────────────────────────────

    @property
    def index_ready(self) -> bool:
        return self._retriever is not None

    @property
    def index_path(self) -> Path:
        return self.cfg.index_dir
