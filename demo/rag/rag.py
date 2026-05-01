"""
demo/rag/rag.py
---------------
Thin shim: replaces the previous Vertex AI Search + Gemini Flash backend
with the local FAISS + BM25 + Groq/Ollama pipeline.

Public API is unchanged — the Flask app calls ask(query) and receives
{"answer": str, "sources": list[dict]} or {"error": str}.

Lazy-initialised on first call so import doesn't fail when the index
hasn't been built yet (e.g. during frontend dev without running build_index).
"""

from pathlib import Path

# Index lives at project root, two levels above this file.
_INDEX_DIR = Path(__file__).parent.parent.parent / "rag" / "index"

_pipeline = None


def _init() -> None:
    global _pipeline
    if _pipeline is not None:
        return

    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline

    cfg = RAGConfig(index_dir=_INDEX_DIR)
    _pipeline = RAGPipeline(config=cfg)
    _pipeline.load_index()


def ask(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → generate.

    Returns:
        {"answer": str, "sources": list[{"chunk_id", "doc_id", "title",
          "source", "text", "rrf_score", "dense_score", "bm25_score"}]}
        or {"error": str} on failure.
    """
    if not query or not query.strip():
        return {"error": "Empty query."}
    try:
        _init()
        return _pipeline.ask(query.strip())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"RAG pipeline error: {exc!s}"[:400]}
