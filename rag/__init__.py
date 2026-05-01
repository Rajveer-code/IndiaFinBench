"""
rag/
----
Local FAISS + BM25 hybrid RAG pipeline for the IndiaFinBench corpus.

Quick start:
    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline()
    pipe.load_index()          # load pre-built index
    result = pipe.ask("What are the KYC norms under SEBI?")
"""

# Lazy import — do not eagerly pull in pipeline here; heavy deps may not be installed.
# Users should import directly: from rag.pipeline import RAGPipeline
