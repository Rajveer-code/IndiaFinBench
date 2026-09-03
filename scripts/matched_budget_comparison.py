"""
matched_budget_comparison.py
------------------------------
Plan v3 Phase 2.5 / cleanup item 1: compares original strict accuracy against
the matched-budget (512-token) re-run's strict accuracy, per model.

BUG FOUND AND FIXED 2026-09-03: this script used to re-score the "original"
side live from evaluation/results/*.csv's prediction column. But that column
is write-time character-truncated (200/300/500 chars depending on model --
see Appendix F.6), while the CSV's own `correct` column was scored on the
FULL untruncated prediction at eval time, before truncation. Re-scoring the
truncated text produced false negatives concentrated on tight-budget models
(9 of 10 models' "original" column silently disagreed with the canonical
evaluation/regime_three_way.json by up to 2.71pp / 11 items on LLaMA-3.3-70B),
which corrupted every Delta-pp in Table 2. Fixed at the source per Guardrail 5:
"original" now comes from regime_three_way.json's strict_406/strict_pct (the
already-correct, full-text score), never re-derived from truncated CSV text.
The matched-budget side is unaffected -- results_matched/*.csv predictions are
written untruncated (Phase 2.1), so live-scoring them is safe and correct.

Reads evaluation/regime_three_way.json (original, canonical) and
evaluation/results_matched/*.csv (new, untruncated, scored live). Only reports
on models present in BOTH (i.e. the rerun completed for that model); prints
which models are still missing rather than silently padding the comparison.

Output: evaluation/matched_budget_comparison.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.evaluate import score_answer  # noqa: E402

REGIME_JSON = Path("evaluation/regime_three_way.json")
MATCHED_DIR = Path("evaluation/results_matched")

# Original per-model completion budget (tokens), from Appendix table_models / F3 -- for
# context in the comparison table only, not recomputed here.
ORIG_BUDGET = {
    "LLaMA-3-8B": 300, "Mistral-7B": 300, "Gemma 3 4B": 512, "GPT-OSS 120B": 512,
    "GPT-OSS 20B": 512, "DeepSeek-R1-Distill": 2048, "LLaMA-3.3-70B": 200,
    "Llama 4 Scout 17B": 1024, "Kimi K2": 512, "Qwen3-32B": 1024, "Gemini 2.5 Flash": 200,
}

# Display Name -> matched-dir filename. The matched dir uses a different, script-local
# key (see matched_budget_rerun.py::model_key) -- map explicitly.
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


def truncation_rate_512(path: Path) -> float:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trunc = sum(1 for r in rows if (r.get("finish_reason") or "").lower() in ("length", "max_tokens"))
    return round(100 * trunc / len(rows), 1) if rows else None


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
    per_model = json.loads(REGIME_JSON.read_text(encoding="utf-8"))["per_model"]
    results = {}
    missing = []
    for label, matched_fname in MATCHED_FILE_MAP.items():
        matched_path = MATCHED_DIR / matched_fname
        entry = per_model.get(label)
        if not matched_path.exists() or entry is None:
            missing.append(label)
            continue
        matched = score_file(matched_path)
        # A CSV that exists but isn't actually finished (still mid-rerun, or a straggler
        # error never retried) must not be silently scored as if it were representative --
        # caught 2026-09-04 when a mid-run Gemini file with 65/160 rows errored still
        # produced a percentage with no warning it covered only 95 of 406 items.
        if matched["n_total"] != 406 or matched["n_errors"] > 0:
            missing.append(f"{label} (incomplete: {matched['n_scored']}/{matched['n_total']} scored, "
                            f"{matched['n_errors']} errors)")
            continue
        orig = {"n_total": 406, "n_scored": 406, "n_errors": 0,
                 "n_correct": entry["strict_406"], "pct": entry["strict_pct"]}
        results[label] = {"original": orig, "matched_budget_512": matched,
                           "delta_pp": round(matched["pct"] - orig["pct"], 2) if matched["pct"] is not None and orig["pct"] is not None else None,
                           "orig_budget": ORIG_BUDGET.get(label),
                           "matched_truncation_pct": truncation_rate_512(matched_path)}
        print(f"{label:<20} original={orig['pct']:>6}%  matched={matched['pct']:>6}%  "
              f"delta={results[label]['delta_pp']:+.2f}pp  (matched errors: {matched['n_errors']})")

    if missing:
        print(f"\nNot yet complete, excluded from comparison: {missing}")

    Path("evaluation/matched_budget_comparison.json").write_text(
        json.dumps({"results": results, "not_yet_complete": missing}, indent=2), encoding="utf-8")
    print(f"\nSaved -> evaluation/matched_budget_comparison.json")

    order = ["LLaMA-3-8B", "Mistral-7B", "Gemma 3 4B", "GPT-OSS 120B", "GPT-OSS 20B",
             "DeepSeek-R1-Distill", "LLaMA-3.3-70B", "Llama 4 Scout 17B", "Kimi K2",
             "Qwen3-32B", "Gemini 2.5 Flash"]
    lines = []
    for label in order:
        if label not in results:
            continue
        v = results[label]
        bold_open, bold_close = ("\\textbf{", "}") if abs(v["delta_pp"]) >= 5 else ("", "")
        lines.append(
            f"{label} & {v['orig_budget']} & {v['original']['pct']:.2f} & "
            f"{v['matched_budget_512']['pct']:.2f} & {bold_open}{v['delta_pp']:+.2f}{bold_close} & "
            f"{v['matched_truncation_pct']:.1f}\\% \\\\"
        )
    table_tex = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Model & Orig.\\ budget & Original & Matched (512) & $\\Delta$pp & Trunc.\\ @512 \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    Path("paper/tables/table_matched_budget.tex").write_text(table_tex, encoding="utf-8")
    print("Wrote paper/tables/table_matched_budget.tex")
    print("REMINDER: copy to paper/tmlr/tmlr_submission/tables/table_matched_budget.tex before compiling.")


if __name__ == "__main__":
    main()
