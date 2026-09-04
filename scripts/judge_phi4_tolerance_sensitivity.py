"""Judge-tolerance sensitivity: rerun the phi4-mini cross-model judge at two
additional numeric-rounding tolerances, alongside the existing full-coverage
1% run already sitting in evaluation/results_judged_phi4/ (that run is reused
as-is -- no need to repeat it).

Requested 2026-09-04 after review flagged the judge rubric's "rounding within
1%" clause as present with no written justification and no sensitivity check.
This produces the missing check: exact numeric match (no tolerance) and a
0.1% tolerance, run under otherwise-identical conditions (same model, same
temperature, same num_ctx sizing, same full REG/NUM/TMP coverage) so the
three tolerance levels are comparable.

Reuses judge_phi4_crossmodel.py's machinery unmodified -- same retry logic,
same context sizing, same per-model checkpointing every 10 items -- by
patching JUDGE_PROMPT's rounding clause and redirecting OUT_DIR/MANIFEST_PATH
per variant before calling its main(). Both variants run sequentially in one
process so only one Ollama request is ever in flight (single local GPU).

Output: evaluation/results_judged_phi4_tol_<variant>/<model_key>_results.csv
        evaluation/judge_run_manifest_tol_<variant>.json
Resumable per variant (judge_phi4_crossmodel.process_model skips ids already
present in that variant's output CSV).
"""
from pathlib import Path

import scripts.scorer_with_judge_gemini as sg

BASE_PROMPT = sg.JUDGE_PROMPT
ROUNDING_LINE = "  * Rounding within 1%\n"
assert ROUNDING_LINE in BASE_PROMPT, "rounding clause text moved -- update this script"

VARIANTS = [
    # (label, prompt_text, out_dir, manifest_path)
    ("exact", BASE_PROMPT.replace(ROUNDING_LINE, ""),
     "evaluation/results_judged_phi4_tol_exact",
     "evaluation/judge_run_manifest_tol_exact.json"),
    ("0.1pct", BASE_PROMPT.replace("Rounding within 1%", "Rounding within 0.1%"),
     "evaluation/results_judged_phi4_tol_0_1pct",
     "evaluation/judge_run_manifest_tol_0_1pct.json"),
]

def main():
    import importlib

    for label, prompt_text, out_dir, manifest_path in VARIANTS:
        print(f"\n{'=' * 60}\nTOLERANCE VARIANT: {label}\n{'=' * 60}")
        sg.JUDGE_PROMPT = prompt_text
        # judge_phi4_crossmodel does `from scripts.scorer_with_judge_gemini import
        # JUDGE_PROMPT` at its own top level; (re-)importing it fresh each loop
        # iteration via importlib.reload picks up the patched sg.JUDGE_PROMPT and
        # recomputes NUM_CTX_INFO (which depends on JUDGE_PROMPT's length) for
        # this variant's actual prompt.
        import scripts.judge_phi4_crossmodel as jpc
        importlib.reload(jpc)

        jpc.OUT_DIR = Path(out_dir)
        jpc.OUT_DIR.mkdir(parents=True, exist_ok=True)
        jpc.MANIFEST_PATH = Path(manifest_path)
        jpc.main()

    print("\nAll tolerance variants complete.")


if __name__ == "__main__":
    main()
