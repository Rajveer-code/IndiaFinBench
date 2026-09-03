"""
matched_budget_comparison.py
------------------------------
Plan v3 Phase 2.5 / cleanup item 1: compares original strict accuracy against
the matched-budget (512-token) re-run's strict accuracy, per model, using the
same four-stage scorer (scripts/evaluate.py::score_answer) for both -- the
budget is the only thing that changed.

Reads evaluation/results/*.csv (original, frozen) and evaluation/results_matched/
*.csv (new). Only reports on models present in BOTH (i.e. the rerun completed
for that model); prints which models are still missing rather than silently
padding the comparison.

Output: evaluation/matched_budget_comparison.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate import score_answer  # noqa: E402
from scripts.novel_methods_utils import MODEL_FILES  # noqa: E402

RESULTS_DIR = Path("evaluation/results")
MATCHED_DIR = Path("evaluation/results_matched")

# MODEL_FILES: "Display Name" -> "xxx_results.csv" (original). The matched dir uses a
# different, script-local key (see matched_budget_rerun.py::model_key) -- map explicitly.
MATCHED_FILE_MAP = {
    "LLaMA-3-8B": "llama_3_8b_results.csv",
    "Mistral-7B": "mistral_7b_results.csv",
    "Gemma 3 4B": "gemma_3_4b_results.csv",
    "GPT-OSS 120B": "gpt_oss_120b_results.csv",
    "GPT-OSS 20B": "gpt_oss_20b_results.csv",
    "DeepSeek-R1-Distill": "deepseek_r1_distill_results.csv",
    "LLaMA-3.3-70B": "llama_33_70b_results.csv",
    "Llama 4 Scout 17B": "llama_4_scout_17b_results.csv",
    "Kimi K2": "kimi_k2_results.csv",
    "Qwen3-32B": "qwen3_32b_results.csv",
    "Gemini 2.5 Flash": "gemini_25_flash_results.csv",
}


def score_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    correct = sum(score_answer(r["ref_answer"], r["prediction"], r["task_type"]) for r in rows
                  if not r.get("error"))
    n_scored = sum(1 for r in rows if not r.get("error"))
    n_err = n - n_scored
    return {"n_total": n, "n_scored": n_scored, "n_errors": n_err,
            "n_correct": correct, "pct": round(100 * correct / n_scored, 2) if n_scored else None}


def main():
    results = {}
    missing = []
    for label, matched_fname in MATCHED_FILE_MAP.items():
        matched_path = MATCHED_DIR / matched_fname
        orig_fname = MODEL_FILES.get(label)
        orig_path = RESULTS_DIR / orig_fname if orig_fname else None
        if not matched_path.exists() or not orig_path or not orig_path.exists():
            missing.append(label)
            continue
        orig = score_file(orig_path)
        matched = score_file(matched_path)
        results[label] = {"original": orig, "matched_budget_512": matched,
                           "delta_pp": round(matched["pct"] - orig["pct"], 2) if matched["pct"] is not None and orig["pct"] is not None else None}
        print(f"{label:<20} original={orig['pct']:>6}%  matched={matched['pct']:>6}%  "
              f"delta={results[label]['delta_pp']:+.2f}pp  (matched errors: {matched['n_errors']})")

    if missing:
        print(f"\nNot yet complete, excluded from comparison: {missing}")

    Path("evaluation/matched_budget_comparison.json").write_text(
        json.dumps({"results": results, "not_yet_complete": missing}, indent=2), encoding="utf-8")
    print(f"\nSaved -> evaluation/matched_budget_comparison.json")


if __name__ == "__main__":
    main()
