# _archive/

This directory preserves intermediate, deprecated, and versioned scripts from the development of IndiaFinBench. None of these are needed to reproduce the paper's results. They are kept for transparency and reproducibility of the development process.

## Contents

| File | Original Location | Purpose | Why Archived |
|------|------------------|---------|--------------|
| `evaluate_new_models.py` | `scripts/` | Evaluated extended model set (v1) | Superseded by `scripts/evaluate_new_models_v2.py` |
| `evaluate_fewshot_cot.py` | `scripts/` | Few-shot chain-of-thought evaluation | Exploratory; not included in paper |
| `evaluate_fewshot_v2.py` | `scripts/` | Few-shot evaluation (v2) | Superseded by v3 |
| `evaluate_fewshot_v3.py` | `scripts/` | Few-shot evaluation (v3) | Supplementary analysis; final results in `evaluation/results_fewshot/` |
| `score_iaa_expansion.py` | `scripts/` | IAA expansion scorer (original) | Superseded by fixed version |
| `score_iaa_expansion_fixed.py` | `scripts/` | IAA expansion scorer (fixed NUM scoring) | Operational script; moved here to keep scripts/ clean |
| `scorer_with_judge_gemini.py` | `scripts/` | LLM-as-judge scoring pipeline | Final judged results in `evaluation/results_judged/` |
| `generate_iaa_expansion.py` | `scripts/` | Generates IAA expansion annotation sheet | One-shot data generation script |
| `patch_docx.py` | `scripts/` | Patches paper .docx with updated tables (v1) | Superseded by patch_docx2.py |
| `patch_docx2.py` | `scripts/` | Patches paper .docx with updated tables (v2) | Paper finalized; no longer needed |
| `produce_paper_v7.py` | `scripts/` | Generates paper v7 markdown | Versioned paper script; final paper is v12 |
| `generate_paper_docx.py` | `scripts/` | Generates paper .docx from markdown | Replaced by direct editing |
| `update_paper_tables.py` | `scripts/` | Updates LaTeX tables in paper | Paper finalized |
| `test_nemotron.py` | `scripts/` | Tests NVIDIA Nemotron model connection | Debugging script |
| `append_batch.py` | `scripts/` | Appends new QA batch to combined JSON | One-shot data pipeline script |
| `save_gemini_annotation.py` | `scripts/` | Saves AI annotator (Gemini) responses | One-shot annotation script |
| `save_ai_annotation.py` | `scripts/` | Saves AI annotator responses | One-shot annotation script |
| `rescore_gemini.py` | `scripts/` | Re-scores Gemini predictions | Used to fix scoring bug; not needed |
| `fix_num_iaa.py` | `scripts/` | Fixes NUM task IAA scoring bug | Applied fix is now in `score_iaa_expansion_fixed.py` |
| `prepare_annotation_batch.py` | `scripts/` | Prepares batches for annotation | One-shot data pipeline script |
| `update_github.py` | `scripts/` | Automates GitHub push | Replaced by manual git workflow |

## How to use canonical scripts

To reproduce the paper results, use:
- `evaluation/evaluate.py` — main evaluation entry point (original 5 models)
- `scripts/evaluate_new_models_v2.py` — extended model evaluations (7 additional models)
- `scripts/evaluate_v7_models.py` — DeepSeek R1 70B, Gemma 4 E4B evaluations
