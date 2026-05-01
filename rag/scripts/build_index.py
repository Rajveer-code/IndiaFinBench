"""
rag/scripts/build_index.py
--------------------------
CLI entry point for the offline index build step.

Usage (from project root):
    python -m rag.scripts.build_index
    python -m rag.scripts.build_index --data-dir data/parsed --index-dir rag/index
    python -m rag.scripts.build_index --chunk-size 800   # ablation: small chunks
    python -m rag.scripts.build_index --chunk-size 2400  # ablation: large chunks

The script exits non-zero on any error so CI/CD pipelines can detect failures.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from rag.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FAISS + BM25 index for the IndiaFinBench RAG pipeline."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory containing rbi/ and sebi/ subdirectories of .txt files.",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=Path("rag/index"),
        help="Output directory for faiss.index, chunks.pkl, bm25.pkl.",
    )
    p.add_argument("--chunk-size",    type=int, default=1600)
    p.add_argument("--overlap",       type=int, default=200)
    p.add_argument("--min-chunk",     type=int, default=100)
    p.add_argument("--batch-size",    type=int, default=64,  help="Embedding batch size.")
    p.add_argument("--device",        type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    # Import here so the script works from any working directory
    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline

    args = parse_args()
    cfg  = RAGConfig(
        data_dir            = args.data_dir,
        index_dir           = args.index_dir,
        target_chunk_chars  = args.chunk_size,
        overlap_chars       = args.overlap,
        min_chunk_chars     = args.min_chunk,
        embedding_batch_size= args.batch_size,
        embedding_device    = args.device,
    )

    logger.info("Config: chunk=%d overlap=%d device=%s", cfg.target_chunk_chars, cfg.overlap_chars, cfg.embedding_device)
    logger.info("Data dir  : %s", cfg.data_dir.resolve())
    logger.info("Index dir : %s", cfg.index_dir.resolve())

    t0 = time.perf_counter()
    try:
        cfg.enable_generation = False
        pipe = RAGPipeline(config=cfg)
        pipe.build_index()
    except Exception:
        logger.exception("Index build failed.")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("Done. Total time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
