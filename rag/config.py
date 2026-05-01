"""
rag/config.py
-------------
Single source of truth for all RAG hyperparameters.
Override via environment variables or by constructing RAGConfig with explicit values.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # ── Chunking ──────────────────────────────────────────────────────────────
    target_chunk_chars: int = 1600   # ~400 tokens at 4 chars/token for English legal text
    overlap_chars:      int = 200    # ~50-token overlap to preserve cross-boundary clauses
    min_chunk_chars:    int = 100    # discard degenerate micro-chunks

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model:      str = "BAAI/bge-base-en-v1.5"  # 768-dim, 512-token max
    embedding_batch_size: int = 64
    embedding_device:     str = "cpu"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k:          int = 5    # final chunks injected into generation context
    candidates:     int = 20   # candidates fetched from each retriever before RRF
    rrf_k:          int = 60   # RRF constant (Cormack et al. 2009 default)
    max_per_source: int = 3    # diversity cap: max chunks from a single "rbi"/"sebi" source

    # ── Generation ────────────────────────────────────────────────────────────
    llm_backend:  str   = field(default_factory=lambda: os.getenv("RAG_LLM_BACKEND", "groq"))
    groq_model:   str   = "llama-3.3-70b-versatile"
    ollama_model: str   = "llama3.2:3b"
    temperature:  float = 0.0    # deterministic; critical for reproducible evaluation
    max_tokens:   int   = 512

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir:  Path = field(default_factory=lambda: Path("data/parsed"))
    index_dir: Path = field(default_factory=lambda: Path("rag/index"))

    def __post_init__(self) -> None:
        self.data_dir  = Path(self.data_dir)
        self.index_dir = Path(self.index_dir)
