"""
rag/models.py
-------------
Canonical data types shared across all RAG modules.
Import from here — never redefine these elsewhere.
"""

from dataclasses import dataclass


@dataclass
class Document:
    doc_id:    str   # filename stem, e.g. "RBI_Master_Dir_084"
    title:     str   # short human-readable label parsed from filename
    source:    str   # "rbi" | "sebi"
    raw_text:  str   # text content; mutated in-place by TextPreprocessor
    file_path: str   # absolute path to source .txt file


@dataclass
class ChunkRecord:
    chunk_id:   str   # f"{doc_id}__{chunk_idx:04d}" — globally unique
    doc_id:     str   # parent Document.doc_id
    title:      str   # inherited from parent Document
    source:     str   # "rbi" | "sebi"
    text:       str   # chunk text as indexed and embedded
    chunk_idx:  int   # 0-indexed position within parent document
    char_start: int   # character offset in preprocessed document text
    char_end:   int   # exclusive end offset


@dataclass
class RetrievalResult:
    chunk:       ChunkRecord
    dense_score: float   # cosine similarity from FAISS  [-1, 1]
    bm25_score:  float   # raw BM25Okapi score           [0, ∞)
    rrf_score:   float   # RRF fused score               (0, 1/30]
    dense_rank:  int     # 1-indexed rank in dense list  (candidates+1 if absent)
    bm25_rank:   int     # 1-indexed rank in BM25 list   (candidates+1 if absent)
