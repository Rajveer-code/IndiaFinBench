"""
rag/scripts/run_evaluation.py
------------------------------
CLI entry point for Phase 3 evaluation.

Stages:
  1. Load frozen index (must exist — run build_index first).
  2. Load or generate the 50-item eval dataset.
  3. Run retrieval-only ablation (B0–B5) — no LLM calls, fast.
  4. Optionally run generation evaluation on B2 (hybrid) and B0 (dense)
     using the configured LLM backend + Gemini faithfulness judge.
  5. Print structured terminal report.
  6. Save JSON report to data/eval/results_<timestamp>.json.

Usage:
    python -m rag.scripts.run_evaluation
    python -m rag.scripts.run_evaluation --no-generation
    python -m rag.scripts.run_evaluation --eval-set data/eval/eval_set.json
    python -m rag.scripts.run_evaluation --configs B0,B2,B5  # subset of ablation
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Phase 3 evaluation for the IndiaFinBench RAG pipeline."
    )
    p.add_argument("--index-dir",    type=Path, default=Path("rag/index"))
    p.add_argument("--data-dir",     type=Path, default=Path("data/parsed"))
    p.add_argument("--eval-set",     type=Path, default=Path("data/eval/eval_set.json"),
                   help="Path to eval_set.json. Generated if missing and --generate-eval set.")
    p.add_argument("--output-dir",   type=Path, default=Path("data/eval"))
    p.add_argument("--no-generation", action="store_true",
                   help="Skip generation evaluation (retrieval metrics only).")
    p.add_argument("--generate-eval", action="store_true",
                   help="Generate synthetic eval set via Gemini if eval_set.json missing.")
    p.add_argument("--n-synthetic",  type=int, default=35,
                   help="Number of synthetic QA pairs to generate.")
    p.add_argument("--configs",      type=str, default=None,
                   help="Comma-separated subset of ablation config IDs to run, "
                        "e.g. B0_dense_only,B2_hybrid,B5_higher_k")
    p.add_argument("--gemini-key",   type=str, default=None,
                   help="Gemini API key (overrides GEMINI_API_KEY env var).")
    p.add_argument("--groq-key",     type=str, default=None,
                   help="Groq API key (overrides GROQ_API_KEY env var).")
    p.add_argument("--max-gen-items",type=int, default=None,
                   help="Limit generation eval to first N items (useful for quick tests).")
    return p.parse_args()


def main() -> None:
    # Force UTF-8 stdout so box-drawing chars in the terminal report work on Windows
    sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    # Apply key overrides before any imports that might read them
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key
    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key

    # ── Imports ───────────────────────────────────────────────────────────────
    from rag.config import RAGConfig
    from rag.data_loader import DataLoader
    from rag.embeddings import BGEEmbedder
    from rag.evaluation import (
        ABLATION_CONFIGS,
        FAITHFULNESS_JUDGE_MODEL,
        FAITHFULNESS_JUDGE_PROMPT_VERSION,
        evaluate_generation,
        load_or_generate_eval_set,
        print_terminal_report,
        run_ablation,
        save_report,
    )
    from rag.bm25_index import BM25Index
    from rag.index import FAISSIndex
    from rag.pipeline import RAGPipeline
    from rag.preprocessing import TextPreprocessor
    from rag.chunking import RecursiveCharacterSplitter

    # ── Load index ────────────────────────────────────────────────────────────
    if not (args.index_dir / "faiss.index").exists():
        logger.error(
            "Index not found at %s. Run: python -m rag.scripts.build_index first.",
            args.index_dir,
        )
        sys.exit(1)

    cfg = RAGConfig(data_dir=args.data_dir, index_dir=args.index_dir)

    logger.info("Loading embedder: %s", cfg.embedding_model)
    t_emb = time.perf_counter()
    embedder = BGEEmbedder(
        model_name  = cfg.embedding_model,
        device      = cfg.embedding_device,
        batch_size  = cfg.embedding_batch_size,
    )
    logger.info("  Embedder loaded in %.1fs (dim=%d)", time.perf_counter() - t_emb, embedder.dim)

    logger.info("Loading FAISS + BM25 index from %s", args.index_dir)
    faiss_idx = FAISSIndex.load(args.index_dir, embedder.dim)
    bm25_idx  = BM25Index.load(args.index_dir)
    logger.info("  Index: %d vectors, %d BM25 chunks", faiss_idx.size, bm25_idx.size)

    # ── Load or generate eval set ─────────────────────────────────────────────
    docs:   list | None = None
    chunks: list | None = None

    if not args.eval_set.exists():
        if not args.generate_eval:
            logger.error(
                "Eval set not found at %s. "
                "Run with --generate-eval to create it, or provide --eval-set path.",
                args.eval_set,
            )
            sys.exit(1)
        logger.info("Generating synthetic eval set (%d items)…", args.n_synthetic)
        loader      = DataLoader(cfg.data_dir)
        preprocessor = TextPreprocessor()
        splitter    = RecursiveCharacterSplitter(
            target_chunk_chars=cfg.target_chunk_chars,
            overlap_chars=cfg.overlap_chars,
            min_chunk_chars=cfg.min_chunk_chars,
        )
        docs   = loader.load()
        for d in docs:
            d.raw_text = preprocessor.process(d.raw_text)
        chunks = [c for d in docs for c in splitter.split_document(d)]

    eval_items = load_or_generate_eval_set(
        path        = args.eval_set,
        docs        = docs,
        chunks      = chunks,
        n_synthetic = args.n_synthetic,
        api_key     = args.gemini_key,
    )
    n_with_gt = sum(1 for i in eval_items if i.relevant_chunk_ids)
    logger.info(
        "Eval set: %d items total (%d with ground-truth chunk IDs, %d adversarial).",
        len(eval_items), n_with_gt, len(eval_items) - n_with_gt,
    )

    # ── Filter ablation configs ───────────────────────────────────────────────
    selected_configs = ABLATION_CONFIGS
    if args.configs:
        ids = {c.strip() for c in args.configs.split(",")}
        selected_configs = [c for c in ABLATION_CONFIGS if c["id"] in ids]
        if not selected_configs:
            logger.error("No matching configs found for: %s", args.configs)
            sys.exit(1)

    # ── Stage 1: Retrieval ablation ───────────────────────────────────────────
    logger.info("Stage 1: Retrieval-only ablation (%d configs)…", len(selected_configs))
    ablation_results = run_ablation(
        base_faiss  = faiss_idx,
        base_bm25   = bm25_idx,
        embedder    = embedder,
        base_cfg    = cfg,
        eval_items  = eval_items,
        configs     = selected_configs,
    )

    # ── Stage 2: Generation evaluation (optional) ─────────────────────────────
    gen_results: dict = {}

    if not args.no_generation:
        gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            logger.warning(
                "GEMINI_API_KEY not set — skipping generation evaluation. "
                "Set it or pass --gemini-key to enable faithfulness scoring."
            )
        else:
            import google.generativeai as genai  # type: ignore[import]
            genai.configure(api_key=gemini_key)
            judge_model = genai.GenerativeModel(
                FAITHFULNESS_JUDGE_MODEL,
                generation_config={"temperature": 0.0, "max_output_tokens": 1024},
            )

            # Run generation eval on B2 (proposed) and B0 (dense baseline)
            for target_id in ("B2_hybrid", "B0_dense_only"):
                target_cfg = next(
                    (c for c in selected_configs if c["id"] == target_id), None
                )
                if target_cfg is None:
                    continue
                logger.info("Stage 2: Generation eval for %s…", target_id)

                # Wire up a full pipeline with the appropriate mode
                pipeline = RAGPipeline(config=cfg)
                pipeline.load_index()

                gm, gf = evaluate_generation(
                    pipeline    = pipeline,
                    eval_items  = eval_items,
                    embedder    = embedder,
                    gemini_model= judge_model,
                    mode        = target_cfg["mode"],
                    max_items   = args.max_gen_items,
                )
                gen_results[target_id] = (gm, gf)

    # ── Print terminal report ─────────────────────────────────────────────────
    print_terminal_report(ablation_results, gen_results, eval_items)

    # ── Save JSON report ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"results_{timestamp}.json"

    config_snapshot = {
        "embedding_model":  cfg.embedding_model,
        "target_chunk_chars": cfg.target_chunk_chars,
        "overlap_chars":    cfg.overlap_chars,
        "top_k":            cfg.top_k,
        "candidates":       cfg.candidates,
        "rrf_k":            cfg.rrf_k,
        "max_per_source":   cfg.max_per_source,
        "llm_backend":      cfg.llm_backend,
        "temperature":      cfg.temperature,
        "index_dir":        str(args.index_dir),
        "eval_items":       len(eval_items),
        "items_with_gt":    n_with_gt,
        "judge_model":      FAITHFULNESS_JUDGE_MODEL,
        "judge_prompt_ver": FAITHFULNESS_JUDGE_PROMPT_VERSION,
        "timestamp":        timestamp,
    }

    save_report(ablation_results, gen_results, config_snapshot, report_path)
    print(f"\n  Report saved → {report_path}")
    print()


if __name__ == "__main__":
    main()
