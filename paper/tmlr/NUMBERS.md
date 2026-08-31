# NUMBERS ledger — IndiaFinBench → TMLR

Every quantitative claim destined for the paper, mapped to the file it comes from.
Regenerated 2026-08-31 against **12 models × 406 items**. Nothing goes in the paper unless it
appears here with a real path.

Paths are relative to `D:\Projects\IndiaFinBench\`.

---

## 0. ⚠ SOURCE-OF-TRUTH CORRECTION — read before using any draft

**`P5_IndiaFinBench_V8_..._Submitted.docx` is NOT what was submitted, and it contains a human
baseline that no data file supports. Do not build the TMLR paper from it.**

| | V8 docx (dated Jul 17) | Submitted PDF / OpenReview | Backed by data? |
|---|---|---|---|
| Human baseline | 69.0%, n=100, CI [59.4, 77.2] | **80.0% (48/60)**, CI [68.2, 88.2] | PDF ✅ / docx ❌ |
| Conclusion | "every model exceeds the human baseline" | "no model performs significantly better" | opposite claims |

**Ground truth, reproduced 2026-08-31** by running `scripts/score_human_eval.py` against
`annotation/human_eval/human_annotator_answer_sheet_filled.csv` (60 rows):

```
Overall: 80.0% (48/60)
REG 100.0% (11/11) | NUM 56.2% (9/16) | CON 82.4% (14/17) | TMP 87.5% (14/16)
```

This matches the submitted PDF **exactly**, including every per-task split.

**No 100-item human study exists.** Exhaustive search of `annotation/`: every human and IAA sheet is
60 items (the IAA study is 3 × 60 = 180), plus 30-item pilots and a 150-item AI-annotator sheet.
Nothing has n=100. The docx's 69.0% and its CI are internally consistent arithmetic over a sample
that has no file behind it.

**Consequence:** the canonical text for the TMLR rebuild is the **submitted PDF**
(`3_IndiaFinBench_An_Evaluation_ (1).pdf`), not the V8 docx. Had the paper been rebuilt from the
docx, it would have published an unsupported number attached to the opposite scientific conclusion.

### RESOLVED 2026-08-31 — two parallel branches, Word vs LaTeX

The paper has two lineages. The upgrade (LLM-judge audit, strict-vs-corrected scoring, the 60-item
human comparison, the measurement framing) was done in **LaTeX**. The Word lineage was never updated
and still carries the pre-upgrade human baseline.

| Lineage | Files | Human baseline | Status |
|---|---|---|---|
| **LaTeX (canonical)** | `IndiaFinBench_ACL_LaTeX_final.zip` → `acl_latex.tex` | 80.0% (48/60) | **submitted, upgraded, data-backed** |
| Word (superseded) | every `*V7*.docx`, `*V8*.docx`, incl. the Aug 6 copy | 69.0%, n=100 | pre-upgrade branch, do not use |

Proof the LaTeX zip is what was submitted: it is dated **2026-07-05**, the OpenReview submission
date, and its bundled `IndiaFinBench_compiled_preview.pdf` is **3,946,974 bytes — the exact byte
size** of `P5_IndiaFinBench_V8_preview_EMNLP2026-FinNLP.pdf` in the submitted folder.

Marker counts in `acl_latex.tex`: `48/60` ×1, `80.0` ×4, `judge` ×22, `strict string-matching` ×4,
`69.0` ×0, `n = 100` ×0.

**The 69.0% figure was not fabricated** — it was an earlier human-baseline design that the upgrade
replaced. It is simply unreproducible today, so it cannot be used.

**Canonical editable source for the TMLR build:**
`paper/tmlr/acl_source/acl_latex.tex` (extracted from the zip, 8,216 words), with `figures/` and
`indiafinbench.bib` alongside it. TMLR conversion is a style swap plus restructure, not a rewrite.

---

## 1. Core results table — VERIFIED against the submitted V8

Recomputed from `evaluation/results/*.csv` via `scripts/novel_methods_utils.get_task_accuracies()`.
**All 60 cells match the submitted PDF exactly.** The headline range is reproducible.

| Model | REG | NUM | CON | TMP | Overall |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | 93.1 | 84.8 | 88.7 | 88.5 | **89.7** |
| Qwen3-32B | 85.1 | 77.2 | 90.3 | 92.3 | 85.5 |
| LLaMA-3.3-70B | 86.2 | 75.0 | 95.2 | 79.5 | 83.7 |
| Llama 4 Scout 17B | 86.2 | 66.3 | 98.4 | 84.6 | 83.3 |
| Kimi K2 | 89.1 | 65.2 | 91.9 | 75.6 | 81.5 |
| LLaMA-3-8B | 79.9 | 64.1 | 93.5 | 78.2 | 78.1 |
| GPT-OSS 120B | 79.9 | 59.8 | 95.2 | 76.9 | 77.1 |
| GPT-OSS 20B | 79.9 | 58.7 | 95.2 | 76.9 | 76.8 |
| Gemini 2.5 Pro | 89.7 | 48.9 | 93.5 | 64.1 | 76.1 |
| Mistral-7B | 79.9 | 66.3 | 80.6 | 74.4 | 75.9 |
| DeepSeek R1 70B | 72.4 | 69.6 | 96.8 | 70.5 | 75.1 |
| Gemma 4 E4B | 83.9 | 50.0 | 72.6 | 62.8 | **70.4** |
| **Task mean** | 83.8 | 65.5 | 91.0 | 77.0 | 79.4 |

Accuracy range **70.4–89.7** — matches the abstract.

Per-model item counts confirmed at 406 with 4 task types for all 12 files.

---

## 2. Regenerated at 12 models (free, post-hoc) — 2026-08-31

| Quantity | Value | Source file |
|---|---|---|
| Kendall's W (rank agreement across tasks) | W=0.853, χ²=30.7, p<0.001, n=12 | `evaluation/novel_methods/kendalls_w/kendalls_w_results.json` |
| Ceiling items | **203 / 406 (50.0%)** | `evaluation/novel_methods/iRT_analysis/discrimination_summary.json` |
| Highly discriminative items | 137 | same |
| Floor items | 6 | same |
| Medium items | 60 | `evaluation/novel_methods/iRT_analysis/item_discrimination.csv` |
| Mean item discrimination | 0.44 | discrimination_summary.json |
| Median item accuracy | 0.875 | same |
| Consensus-hard items (<20% models correct) | 11 | `exp11` stdout; `evaluation/novel_methods/era_stratification/` |
| Consensus-easy items (>90% correct) | 203 | same — **independent cross-check of the ceiling count** |
| ~~Max dissociation model~~ | **CUT — see §5 metric-naming** | derivable from Table 2; reads as manufactured novelty |
| CON label balance | 9 Yes / 53 No, majority baseline 85.48% | `evaluation/novel_methods/con_balance/con_balance_summary.json` |
| Models above CON majority baseline | **10 of 12** (recomputed) | direct computation — `exp3` stdout says 8, a **script counting bug**; see §5 |
| TMP items by complexity tier | High 40 / Medium 28 / Low 10 (n=78) | `evaluation/novel_methods/tmp_depth/exp1_summary.json` |
| TMP complexity↔accuracy correlation | **no model significant** (12/12 n.s.) | same |
| Top difficulty predictor | flesch_ease, coef=0.708 | `evaluation/novel_methods/feature_regression/feature_importance.csv` |
| Context length ↔ accuracy | all \|r\|≤0.087, none significant | `evaluation/novel_methods/context_length/context_length_correlations.csv` |

### Values that CHANGED when the 12th model was added

Anything quoting the old figures is stale. Do not reuse.

| Quantity | 11 models (old) | 12 models (current) |
|---|---|---|
| Kendall's W | 0.841 (χ²=27.764) | **0.853** (χ²=30.7) |
| Ceiling items | 215 | **203** |
| Highly discriminative | 116 | **137** |
| Floor items | 7 | **6** |
| Mean discrimination | 0.427 | **0.44** |
| Median item accuracy | 0.909 | **0.875** |
| Max dissociation | DeepSeek R1 70B, 0.157 | **Gemini 2.5 Pro, 0.1868** |
| Mean DI | 0.0749 | **0.0842** |

---

## 3. FROZEN at 11 models — cannot be regenerated without paid inference

These scripts call `call_gemini` / `call_groq` and would re-run inference. Verified call sites:
`exp2:133`, `exp4:102`, `exp9:158,165`, `exp10:67,69`.

Backup of the original outputs: `evaluation/_novel_methods_BACKUP_11models_2026-08-31/`

| Analysis | Coverage | Source | Usable? |
|---|---|---|---|
| RSTS metric (exp2) | 11 models, subset | `novel_methods/rsts_scores/` | Only if the subset is stated explicitly |
| Scoring audit (exp4) | 100 items, FN rate 35%, implied correction +17.5% | `novel_methods/scoring_audit/scoring_audit_summary.json` | Yes, with stated scope |
| RAG evaluation (exp9) | **120 items, 1 model** — badly incomplete | `novel_methods/rag_evaluation/rag_results_partial.csv` | **No — omit or redo locally** |
| Perturbation (exp10) | 11 models | `novel_methods/perturbation/` | Only if the subset is stated explicitly |

**Scoring-audit caution:** exp4 reports a 35% false-negative rate over 100 audited items. The
submitted paper reports a **78.4%** judge flip rate over 874 flagged predictions and a 28-item manual
audit at 89.3% agreement. These are different populations and different procedures — if both appear
in the paper, the distinction must be explicit or a reviewer will read it as a contradiction.

---

## 3b. Scoring-regime comparison — VERIFIED 2026-08-31, this is the hero result

Computed from `evaluation/results_judged/*.csv` (`correct` column = item-level judge-corrected
verdict; `auto_score` = strict pipeline) against strict accuracies from `evaluation/results/`.

| Model | Strict | Rank | Corrected | Rank | Shift |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | 89.7 | 1 | 95.6 | 5 | −4 |
| Qwen3-32B | 85.5 | 2 | 95.8 | 4 | −2 |
| LLaMA-3.3-70B | 83.7 | 3 | 95.6 | 6 | −3 |
| Llama 4 Scout 17B | 83.3 | 4 | 96.1 | 2 | +2 |
| Kimi K2 | 81.5 | 5 | 96.1 | 3 | +2 |
| LLaMA-3-8B | 78.1 | 6 | 91.4 | 10 | −4 |
| GPT-OSS 120B | 77.1 | 7 | 95.6 | 7 | 0 |
| GPT-OSS 20B | 76.8 | 8 | 93.8 | 9 | −1 |
| Gemini 2.5 Pro | 76.1 | 9 | 95.1 | 8 | +1 |
| Mistral-7B | 75.9 | 10 | 90.6 | 11 | −1 |
| **DeepSeek R1 70B** | **75.1** | **11** | **96.6** | **1** | **+10** |
| Gemma 4 E4B | 70.4 | 12 | 79.8 | 12 | 0 |

- **Corrected range: 79.8 – 96.6** ✅ matches the submitted abstract exactly.
- **DeepSeek R1 70B moves 11th → 1st**, a +10 position shift.
- DeepSeek errors: **101 strict → 14 corrected; 87 reclassified = 86.1%** ✅ matches the "86%" claim.
- **Rank concordance between regimes: Spearman ρ = 0.455 (p = 0.1377); Kendall τ = 0.364 (p = 0.1160).**

**Wording discipline for ρ:** n = 12 gives low power. State as "the strict ranking explains little of
the corrected ranking, and the association is not statistically significant at this sample size."
Do **not** write "no correlation" or "uncorrelated" — failing to reject H₀ is not evidence of absence.

**Note on the superseded Word branch:** its Table D1 reported corrected values of 80.8–93.3 derived
from a *formula* (`corrected = reported + FNR × (1 − reported)`). The LaTeX branch uses actual
item-level judge verdicts, which is more rigorous and yields 79.8–96.6. Use the LaTeX method.

---

## 4. Numbers still to verify before use (from V8, not yet re-derived)

- [ ] Judge-corrected accuracy bounds 79.8–96.6% → `evaluation/results_judged/`
- [ ] Judge flip rate 78.4% (685/874) and per-task splits
- [ ] DeepSeek R1 70B: 86% of errors reclassified as format non-compliance
- [ ] Bootstrap p-values: Llama 4 Scout vs LLaMA-3.3-70B p=0.80; GPT-OSS 120B vs 20B p=0.861;
      Gemini 2.5 Flash vs Qwen3-32B p=0.041 → `evaluation/bootstrap_significance_results.json`
- [ ] Wilson CIs incl. human 80.0% (48/60) [68.2, 88.2] → `evaluation/wilson_ci_results.json`
- [ ] IAA: κ=0.645 CON, 77.2% overall human, 90.7% model-based
- [ ] Dataset composition: 406 items, 192 docs, REG 174 / NUM 92 / CON 62 / TMP 78;
      difficulty 160 easy / 182 medium / 64 hard

---

## 5. Significance analysis — REGENERATED 2026-08-31 (was stale at 11 models)

`evaluation/bootstrap_significance_results.json` held **55 pairs (11 models)** — Gemini 2.5 Pro was
missing, the same root cause as the `novel_methods` outputs. The paper nonetheless claimed 66 pairs.

**Root cause:** two hardcoded model lists in `scripts/generate_figures.py` omitted Gemini 2.5 Pro —
`MODEL_FILES` (line 52) and `short_names` (line 334). Both fixed; backup at
`scripts/generate_figures.py.bak`. Old JSON backed up to
`evaluation/_bootstrap_11models_BACKUP_2026-08-31.json`.

| Quantity | Paper (V8) | Verified 12-model | Verdict |
|---|---|---|---|
| Model pairs | 66 | **66** | ✅ now correct |
| Pairs significant at p<0.05 | 42 | **41** | ❌ fix to 41 |
| Bonferroni threshold | 0.05/66 ≈ 0.00076 | 0.00076 | ✅ |
| Pairs surviving Bonferroni | 19 | **19** | ✅ |
| LLaMA-3.3-70B vs Llama 4 Scout | p = 0.790 | **p = 0.7933** | fix to 0.793 |
| GPT-OSS 120B vs 20B | p = 0.910 / 0.861 | **p = 0.9071** | both wrong; use 0.907 |
| Gemini 2.5 Flash vs Qwen3-32B | p = 0.057 / 0.041 | **p = 0.0574** | use 0.057 |

Gemini 2.5 Pro is significantly below all five Tier-1 models and significantly *above* Gemma 4 E4B
(p = 0.0172); it is statistically indistinguishable from the whole Tier-2 group.

### Other verified corrections

| Claim | Status |
|---|---|
| CON: "Ten of twelve exceed the 85.5% majority baseline" | ✅ **correct** (recomputed: 10 of 12). `exp3`'s printed "Models above this: 8" is a **counting bug in the script**, not a data error. |
| Human subset: Gemini 2.5 Flash 85.0%, Gemma 4 E4B 63.3%, human 80.0% | ✅ verified by direct recomputation over the 60 item IDs; 5 of 12 models score above the evaluator |

### Metric-naming corrections (overclaim risk)

| Artifact | Problem | Action |
|---|---|---|
| `exp7` / `iRT_analysis/` | Labelled "IRT-lite" but fits **no item-response model**; computes $s=4p(1-p)$, a normalised variance | **Never call it IRT.** Drafts use "spread" / "effective size" |
| `exp8` Dissociation Index | $DI=(CON-TMP)/(CON+TMP)$ — a renamed gap between two columns already in Table 2 | **Cut.** Adds no information; reads as manufactured novelty |
| `exp8` Jaccard error overlap | Genuine added information | **Keep.** GPT-OSS 20B/120B overlap 0.427, highest of all 66 pairs (median 0.270) |

### ⚠ Figures — outstanding

`paper/figures/figure1_heatmap.png` etc. are dated **2026-04-21** (11 models). The regenerated
12-model versions are written under different filenames (`performance_heatmap.png`,
`radar_chart.png`). **Phase C must confirm whether `acl_source/figures/*.png` (dated 2026-07-03)
contain 11 or 12 models** — if 11, the submitted paper's figures disagree with its own text.

---

## 6. Figure audit — 2026-08-31

### The submitted paper contradicted itself

`acl_source/figures/fig_heatmap.png` (dated 2026-07-03) shows **11 models**. Gemini 2.5 Pro is
absent. The manuscript text describes it as "across all twelve models." Confirmed by direct visual
inspection of the image, not inferred. The desk rejection meant no reviewer ever saw it.

### Regenerated and verified

All four data figures regenerated at 300 dpi and checked cell-by-cell against
`get_task_accuracies()`:

| Figure | Models | Verification |
|---|---|---|
| `figure1_heatmap.png` | **12** ✅ | 48/48 cells match; **now data-driven** |
| `figure2_radar.png` | **12** ✅ | 0 mismatches across all 12 models |
| `figure3_difficulty.png` | **12** ✅ | 0 mismatches; item counts 160/182/64 confirmed from data |
| `figure4_correlation.png` | **12** ✅ | NUM–TMP $\rho = 0.691$, $p = 0.0128$ — the only significant pair, as the caption claims |

### Reproducibility fix

`paper/figures/generate_heatmap.py` hardcoded all 48 accuracy values. Every one was correct, but a
hardcoded figure can silently drift from the data. It now loads from
`novel_methods_utils.get_task_accuracies()`. Backup: `generate_heatmap.py.bak`.

**Still hardcoded** (values verified correct today, but same drift risk):
`generate_radar.py`, `generate_difficulty_plot.py`, `generate_correlation.py`.

### Open decisions for Phase C

1. The regenerated heatmap **drops the "Overall" column and the per-task item counts** that the
   submitted version carried. Either restore them or accept the simpler figure.
2. **Consider cutting the heatmap altogether.** It renders the same 48 numbers as Table 2 — a figure
   restating a table is the same redundancy the prose rules forbid. The space is better spent on a
   figure of the hero result: a rank-shift plot of strict vs corrected ordering, showing DeepSeek R1
   70B moving 11th → 1st. No such figure exists yet; it would have to be written.

---

## 7. Scoring-regime table + rank-shift figure — 2026-08-31

Generated by `scripts/generate_regime_figure.py` from `evaluation/results_judged/*.csv`.
Nothing hardcoded. Outputs `paper/figures/figure_regime_shift.{png,pdf}` (300 dpi) and
`paper/tables/table_regime.tex`. Asserts 12 models × 406 items before writing anything.

| Model | Strict | Rank | Corrected | Rank | ΔRank |
|---|---|---|---|---|---|
| Gemini 2.5 Flash | 89.7 | 1 | 95.6 | 5 | −4 |
| Qwen3-32B | 85.5 | 2 | 95.8 | 4 | −2 |
| LLaMA-3.3-70B | 83.7 | 3 | 95.6 | 5 | −2 |
| Llama 4 Scout 17B | 83.3 | 4 | 96.1 | 2 | +2 |
| Kimi K2 | 81.5 | 5 | 96.1 | 2 | +3 |
| LLaMA-3-8B | 78.1 | 6 | 91.4 | 10 | −4 |
| GPT-OSS 120B | 77.1 | 7 | 95.6 | 5 | +2 |
| GPT-OSS 20B | 76.8 | 8 | 93.8 | 9 | −1 |
| Gemini 2.5 Pro | 76.1 | 9 | 95.1 | 8 | +1 |
| Mistral-7B | 75.9 | 10 | 90.6 | 11 | −1 |
| **DeepSeek R1 70B** | **75.1** | **11** | **96.6** | **1** | **+10** |
| Gemma 4 E4B | 70.4 | 12 | 79.8 | 12 | 0 |

Corrected values independently reproduce the abstract's **79.8–96.6** range. Strict column matches
§1 exactly.

**Competition ranking is required.** Three models tie at 95.6 and two at 96.1. Ranking without tie
handling invents distinctions the data does not support — an earlier draft did exactly that and
reported ΔRank wrong for LLaMA-3.3-70B, Kimi K2, and GPT-OSS 120B.

### Corrections to earlier drafts

| Claim | Draft said | Verified |
|---|---|---|
| Spearman strict vs corrected | ρ = 0.455, p = 0.14 | **ρ = 0.413, p = 0.1824** |
| Kendall τ | 0.364, p = 0.12 | **0.313, p = 0.1655** |
| Models moving ≥2 ranks | six of twelve | **eight of twelve** |

### New finding — the correction compresses the field

Excluding Gemma 4 E4B (the only model whose errors survive the audit in bulk), the remaining eleven
models span **14.6 pp** strict (75.1–89.7) and **6.0 pp** corrected (90.6–96.6). Format compliance
accounts for roughly two thirds of the apparent spread between mid-table models. Now in the abstract
and §regime.

**Note:** `evaluation/results_judged/pipeline_comparison.csv` disagrees slightly (Gemini 2.5 Flash
89.9 at n=405, Qwen3-32B 85.9 at n=404) because it was computed over subsets. The per-item
recomputation over all 406 rows matches §1 exactly and is authoritative.

---

## 8. MAJOR FINDING — Gemma was mislabelled; corrected model reshuffles the leaderboard (2026-08-31)

The original `evaluation/results/gemma4_e4b_results.csv` was produced by a script requesting Ollama
tag `"gemma4"` — not a real Google release; no surviving log confirms what it actually resolved to.
Its own `hf_id` field read `google/gemma-4-e4b`, a model Google never released. Re-run from scratch
on **`gemma3:4b`**, confirmed via `/api/show`: family `gemma3`, **4,299,915,632 params**, Q4_K_M —
matching the paper's own methods text and the real `google/gemma-3-4b` release.

| Task | Old (`gemma4`, ambiguous) | New (`gemma3:4b`, confirmed) | Δ |
|---|---|---|---|
| REG | 83.9 | 86.2 | +2.3 |
| NUM | 50.0 | 70.7 | **+20.7** |
| CON | 72.6 | 79.0 | +6.5 |
| TMP | 62.8 | 71.8 | +9.0 |
| **Overall** | **70.4** | **78.8** | **+8.4** |

**Gemma moves from 12th of 12 (isolated "Tier 3," the paper's floor) to 6th of 12** — now above
LLaMA-3-8B, both GPT-OSS variants, Gemini 2.5 Pro, Mistral-7B, and **DeepSeek-R1-Distill-Llama-70B**.

**Every claim keyed to Gemma being the weakest/outlier model is now false and must not be reused:**
- Abstract range "70.4% (Gemma 4 E4B) to 89.7%" — floor is now DeepSeek at 75.1%
- "Gemma 4 E4B stands alone" / Tier 3 framing — gone; Gemma is mid-pack
- "near-chance numerical reasoning" for Gemma (was 50.0%) — now 70.7%, nowhere near chance
- "only Gemma is significantly worse than the human baseline" — must be recomputed
- Difficulty-collapse claim ("82.5%→56.2%") — must be recomputed from the new predictions
- The abstract's compression finding ("excluding Gemma, remaining eleven span...") — the premise
  that Gemma is the outlier no longer holds; must be recomputed or reframed

Backup of the ambiguous-identity original:
`evaluation/results/_gemma4_e4b_results_AMBIGUOUS-IDENTITY_BACKUP_2026-08-31.csv`.
New rerun driver: `scripts/rerun_gemma3_4b.py`.

### Full pipeline refresh triggered by this fix

Re-ran, in order: the 8 free `evaluation/novel_methods/` analyses (second time — their first
12-model run, committed earlier today, still used the ambiguous Gemma file), then the canonical
`scripts/generate_figures.py` (writes `bootstrap_significance_results.json`,
`wilson_ci_results.json`, `difficulty_breakdown.csv`, `task_accuracy_matrix.csv`).

**Bootstrap significance: 36 of 66 pairs significant at p<0.05** (was 41 with the wrong Gemma, 42 in
the stale submitted paper). Fewer significant pairs is the expected, correct consequence of Gemma no
longer being a dramatic outlier — it now sits close enough to several mid-table models that fewer
comparisons clear p<0.05.

**Caution — a wrong tool was nearly used here.** `scripts/bootstrap_significance.py` (legacy,
separate from `generate_figures.py`) outputs to `evaluation/error_analysis/` and includes a phantom
**"Claude 3 Haiku"** entry that is not one of the 12 benchmarked models and was evaluated (per
`evaluate_v7_models.py`'s dead code) on a different 150-item subset. Running it was reverted via
`git checkout` before anything was staged. **Do not use `scripts/bootstrap_significance.py` or
anything under `evaluation/error_analysis/` for this paper — `generate_figures.py` is the canonical
source for `bootstrap_significance_results.json` and `wilson_ci_results.json`.**

### Still stale, not yet fixed

`evaluation/results_judged/gemma4_e4b_results.csv` — the Gemini-as-judge audit of Gemma's OLD,
wrong predictions. Superseded by the Phase 2 full-coverage local judge (§9) rather than patched,
since the judge methodology itself was being replaced regardless.

---

## 9. Phase 2 — cross-model judge (phi4-mini), full coverage

Closes the objection that the original judge (Gemini 2.5 Flash) is also one of the 12 evaluated
models. `phi4-mini` (Microsoft Phi, 3.8B, local via Ollama) shares no family with any benchmarked
model and costs nothing to run.

**Scope: all REG/NUM/TMP items for all 12 models = 344 × 12 = 4,128 judged predictions** — full
coverage, not only strict failures, so this is the first point at which a strict **false-positive**
rate becomes estimable. CON excluded, matching the paper's existing rationale (exact Yes/No against
an unambiguous binary label; no semantic review needed) — a pre-existing principled choice, not a
new gap.

Reuses `JUDGE_PROMPT` and `JUDGE_TASKS` verbatim from `scripts/scorer_with_judge_gemini.py` — same
rubric, different judge, so the standard being applied doesn't change, only who applies it.

**Validated before committing to the full run:**
- 8-item pilot on Gemini's own strict-correct REG items: 8/8 judge-agrees (sanity check, not
  informative on its own).
- 8-item stress test on DeepSeek-R1-Distill-Llama-70B's strict-*incorrect* REG items: **7/8 flipped
  to correct** (genuine format mismatches — digit vs. written-out numbers, extra preamble text before
  the answer) and **1/8 correctly stayed incorrect** (REG_003: predicted "ten per cent" against a
  reference of "twenty per cent" — a real error). This confirms the judge discriminates rather than
  rubber-stamping either the strict score or a blanket "correct".
- Timing: 3.29s/call measured → **projected ~226 min (≈3.75h)** for full coverage.

**Launched in background** (`scripts/judge_phi4_crossmodel.py` → `evaluation/results_judged_phi4/`),
checkpointing every 10 items per model. Output columns: `id, task_type, difficulty, question,
ref_answer, prediction, strict_correct, judge_verdict, judge_reason`.

**Decision gate (committed before results are seen, per the approved plan):** if the phi4-mini
judge supports DeepSeek's strict-11th → corrected-1st reversal, it stays as the hero result. If not,
the paper's spine becomes the discriminative-coverage / compression finding instead.

**Known limitation, to be stated plainly in the paper:** phi4-mini is a 3.8B model. Disagreement
with Gemini's verdicts is ambiguous between "Gemini is biased" and "phi4-mini lacks the capacity for
this task" — only human adjudication (oversampling exactly these disagreement cases) can resolve
which. That adjudication sheet is built once the full phi4-mini pass completes, so disagreement
cases are known.

**Correction to the master plan's own estimate:** the plan said "4,872 predictions (12×406)". The
real full-coverage scope under the existing JUDGE_TASKS design is **4,128** (12×344, CON excluded),
not 4,872.

---

## 10. Phase 1 — provenance audit and document-clustered significance (2026-08-31)

`scripts/provenance_and_clustered_bootstrap.py`. Confirms and quantifies the finding in §8: 34
documents represented (not 192 — that's the collected corpus), connected-component clustering
(merging documents co-linked by any CON item) yields **27 independent clusters**.

| Quantity | Value |
|---|---|
| Documents collected (corpus) | 192 (92 SEBI + 100 RBI) |
| Documents represented in the 406 QA items | **34** |
| Connected components (resampling units) | **27** |
| Largest component | 68 items |
| Median component | 7 items |
| Components of size 1 | 4 |

**Correctness checks passed:** clusters disjoint and exhaustive (every item in exactly one, no
document in two components); fixed-seed determinism confirmed (identical output across two runs).
Note: "clustered CI must be wider than item-level" is **not** a valid check and was not used — no
such mathematical guarantee exists.

**Headline comparisons — item-level vs document-clustered bootstrap, all six AGREE in
significance:**

| Pair | Item p | Cluster p | Conclusion |
|---|---|---|---|
| Gemini 2.5 Flash vs Qwen3-32B | 0.0517 | 0.0600 | agree (n.s.) |
| Gemini 2.5 Flash vs Gemma 4 E4B | 0.0000 | 0.0000 | agree (sig.) |
| Llama 4 Scout 17B vs LLaMA-3.3-70B | 0.7878 | 0.8251 | agree (n.s.) |
| GPT-OSS 120B vs GPT-OSS 20B | 0.9037 | 0.9355 | agree (n.s.) |
| Gemma 4 E4B vs DeepSeek R1 70B | 0.1856 | 0.4242 | agree (n.s.) |
| Gemma 4 E4B vs Mistral-7B | 0.2540 | 0.4307 | agree (n.s.) |
| **Human vs Gemini 2.5 Flash (60-item subset)** | 0.4272 | 0.5581 | **agree (n.s.)** |

**This is a genuine strength for the paper, not a caveat to bury.** Clustering did not overturn a
single headline conclusion — the human-comparison claim, the parameter-efficiency null results, and
the significance/non-significance pattern all survive the independence correction. State this
explicitly rather than only in a robustness appendix.

Outputs: `evaluation/provenance_audit.csv`, `evaluation/provenance_summary.json`,
`evaluation/clustered_bootstrap.json`.

---

## 11. Item-discrimination counts — THIRD correction (2026-08-31, same day)

`exp7`'s post-Gemma-fix rerun (committed in §8's pipeline-refresh commit) gives **213 ceiling / 134
discriminative / 7 floor** — not 203/137/6, which was itself a second-generation number computed
between the two Gemma reruns. Cross-checked two ways: `exp11`'s independent consensus count now
also reads 213, and a fresh standalone script (`scripts/threshold_sensitivity_and_dedup.py`)
reproduces 213/7/134 exactly from raw correctness data. Fixed in all four drafts
(01/02/03/06) — verified zero remaining occurrences of "203" or "137".

Also fixed: median item accuracy 91.7% (was 87.5%), mean spread 0.419 (was 0.44), Intermediate band
52 items (was 60).

Threshold-sensitivity table (Phase 3.3) at `evaluation/threshold_sensitivity.json`: ceiling ranges
213 (p≥0.85 or p≥0.90, identical) to 143 (p≥0.95); discriminative band ranges 83 to 168 across three
reasonable cutoffs. The paper's chosen cutoffs sit mid-range, not at an extreme — evidence the
number wasn't picked to maximise the finding.

## 12. ⚠ PENDING — do not patch these again until the phi4-mini judge finishes

The following are known-stale RIGHT NOW and will change again once `evaluation/results_judged_phi4/`
completes (background job, launched with ETA ~3.75h from §9's write-up). Patching them before then
is wasted work:

- **Spearman ρ = 0.455, p = 0.14** in `draft_02_introduction.tex` line ~45 — stale from before even
  the §7 correction (should already be 0.413/0.18 per §7, was missed in an earlier pass, will change
  a THIRD time once Gemma's corrected accuracy enters the regime table under the new judge).
- **The entire strict-vs-corrected regime table and `figure_regime_stat.png`** — built from
  `evaluation/results_judged/` (Gemini-as-judge, and for Gemma specifically, audits the OLD wrong
  predictions). Must be rebuilt from `evaluation/results_judged_phi4/` once complete, using
  `scripts/generate_regime_figure.py`'s pattern adapted to the new judge's column names
  (`strict_correct`, `judge_verdict` instead of `auto_score`, `judge_score`).
- **The abstract's "excluding Gemma, remaining eleven span 14.6/6.0 points" compression finding** —
  its entire premise (Gemma is the outlier to exclude) no longer holds now that Gemma is 6th of 12.
  Needs either recomputing with whichever model is now the true outlier, or reframing entirely.
- **The decision gate on DeepSeek's 11th→1st reversal** (committed in §9) has not yet been
  evaluated — it fires once the phi4 pass completes.

**Action when the judge finishes:** rebuild the regime table/figure first, re-derive ρ/τ and the
compression claim from that, apply the decision gate, then do one clean sweep of every draft file.
Do not touch these numbers piecemeal before that point.

---

## 13. Full post-Gemma-fix sweep of draft_05/06 (2026-08-31) — strict-side only

All strict-side (judge-independent) numbers in `draft_05_results_detail.tex` and
`draft_06_discussion_conclusion.tex` corrected in one pass:

| Claim | Was | Now | Source |
|---|---|---|---|
| Overall range | 70.4–89.7 (19.3pp) | **75.1–89.7 (14.6pp)** | direct computation |
| CON, Gemma | 72.6% | **79.0%** | `evaluation/results/gemma4_e4b_results.csv` |
| Human-subset, Gemma | 63.3% | **66.7%**, still sig. worse (p=0.020) | direct + clustered bootstrap |
| Human-subset, best model p-value | (unstated) | **p=0.43** (item-clustered, §10) | `evaluation/clustered_bootstrap.json` pattern |
| Bonferroni survivors | 19 of 66 | **15 of 66** | `evaluation/bootstrap_significance_results.json` |
| Scout–70B / GPT-OSS pair p (3dp) | 0.793 / 0.907 | **0.786 / 0.912** | same file, precision fix |
| Jaccard median | 0.270 | **0.274** | `error_jaccard_similarity.csv` (rerun with fixed Gemma) |
| DeepSeek naming | "DeepSeek R1 70B" | **"DeepSeek-R1-Distill-Llama-70B"**, short form "DeepSeek-R1-Distill" after first use | `model_version` field is `deepseek/deepseek-r1-distill-llama-70b` |

### Structural finding, not just a number change

The difficulty-stratification subsection claimed "most models decline monotonically, as intended,"
illustrated by a since-corrected Gemma "26.3-point collapse." **Recomputed from the fixed data: only
3 of 12 models decline monotonically at all** (Gemini 2.5 Flash, Gemini 2.5 Pro, GPT-OSS 120B).
Six models score *higher* on hard items than medium; Gemma and GPT-OSS 20B dip at medium and
partially recover on hard. The subsection was rewritten, not patched — the dominant pattern in the
data is not the one the paragraph originally described. New framing ties this to the effective-size
argument already in the paper: nominal difficulty and empirical difficulty are different quantities,
the same relationship as nominal vs. discriminative size.

**Deliberately still stale (see §12):** the regime table, ρ/τ, and the "excluding Gemma" compression
claim in `draft_01`/`draft_03`. Do not touch until the phi4-mini judge completes.
