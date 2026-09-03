"""
matched_budget_correlation.py
------------------------------
Cleanup pass, 2026-09-03: Appendix G.2 reported a Pearson correlation of
r=0.09 ("essentially zero") between each model's original write-time-cap rate
and its matched-budget delta, to argue logging truncation doesn't obviously
explain the deltas. That number was computed against the pre-fix deltas from
matched_budget_comparison.py, which had a bug (see that script's docstring --
the "original" side was live-rescored against write-time-truncated CSV text,
corrupting most deltas). Recomputes the same correlation against the
corrected deltas in evaluation/matched_budget_comparison.json.

Output: evaluation/matched_budget_correlation.json
"""
import json
from pathlib import Path

import numpy as np

trunc = json.loads(Path("evaluation/truncation_disclosure.json").read_text(encoding="utf-8"))["per_model_caps"]
comparison = json.loads(Path("evaluation/matched_budget_comparison.json").read_text(encoding="utf-8"))["results"]

# comparison label -> truncation_disclosure key
LABEL_TO_TRUNC_KEY = {
    "LLaMA-3.3-70B": "groq70b", "GPT-OSS 120B": "gpt_oss_120b", "GPT-OSS 20B": "gpt_oss_20b",
    "Kimi K2": "kimi_k2", "LLaMA-3-8B": "llama3", "Llama 4 Scout 17B": "llama4scout",
    "Mistral-7B": "mistral", "Qwen3-32B": "qwen3_32b", "Gemma 3 4B": "gemma4_e4b",
    "Gemini 2.5 Flash": "gemini",
    # DeepSeek-R1-Distill deliberately excluded -- it's the outlier being investigated,
    # not one of the "other models" the correlation is computed across.
}

pairs = {}
for label, trunc_key in LABEL_TO_TRUNC_KEY.items():
    pairs[label] = (trunc["deepseek_r1_70b" if False else trunc_key]["pct_at_cap"],
                     comparison[label]["delta_pp"])

xs = np.array([v[0] for v in pairs.values()])
ys = np.array([v[1] for v in pairs.values()])
r = float(np.corrcoef(xs, ys)[0, 1])

result = {"n": len(pairs), "pearson_r": round(r, 4), "pairs": pairs}
Path("evaluation/matched_budget_correlation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

print(f"n = {len(pairs)}")
print(f"Pearson r (at-cap %% vs matched-budget delta) = {r:.4f}")
for label, (x, y) in pairs.items():
    print(f"  {label:22} at_cap={x:>5.1f}%  delta={y:+.2f}pp")
print("\nSaved -> evaluation/matched_budget_correlation.json")
