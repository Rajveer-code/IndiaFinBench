# Paper map + TMLR restructure plan

Step 0 of the manuscript-flow pass. Built from measured section lengths
(`analyze_tex.py`) over the canonical `acl_source/acl_latex.tex`.
**Nothing is rewritten until this map is confirmed.**

## 0. Assets

| Asset | Location |
|---|---|
| Manuscript source | `paper/tmlr/acl_source/acl_latex.tex` (8,216 words raw; **3,024 words main body**, 706 appendix) |
| Figures | `acl_source/figures/` — 6 PNGs (pipeline, taskexamples, heatmap, correlation, difficulty, radar) |
| Bibliography | `acl_source/indiafinbench.bib` |
| Results | `evaluation/results/`, `evaluation/novel_methods/`, `evaluation/results_judged/` |
| Number ledger | `paper/tmlr/NUMBERS.md` |

## 1. Central claim — one sentence

> Benchmark accuracy conflates what a model knows with what a format-constrained scoring pipeline
> can extract from it; measuring both regimes reorders the leaderboard, and item-level analysis
> shows half the benchmark carries no discriminative signal at all.

The Indian regulatory benchmark is the **instrument** that makes this measurable, not the claim.

## 2. Canonical order — definition order governs reporting order everywhere

**Task axes (unchanged, already consistent):** REG → NUM → CON → TMP.
Defined §3.2 in this order; Tables 2/7/D1 report in this order. Keep.

**Measurement layers (new spine — abstract, contributions, results, discussion all follow this):**

1. **Strict** — automated four-stage string-matching pipeline
2. **Corrected** — item-level LLM-judge audit of every flagged error
3. **Discriminative** — item-level analysis of which items separate models at all
4. **Human** — a non-specialist reference point on the hardest subset

## 3. Hero result — the abstract headlines ONE finding

> The scoring regime, not the model, decides the ranking: DeepSeek R1 70B places 11th of twelve
> under strict extractive scoring, yet the judge attributes 86% of its errors to format
> non-compliance rather than incorrect reasoning.

Everything else is supporting evidence. Demoted from co-equal headline status: the parameter-
efficiency pairs, the tier structure, and the human comparison.

## 4. Section inventory (measured)

| Section | Words | Verdict |
|---|---|---|
| Abstract | 71 | Rewrite — resize human claim, lead with hero result |
| 1 Introduction | 395 | Reframe to measurement spine |
| 2.1 Financial NLP Benchmarks | 130 | Keep, trim |
| 2.2 Regulatory & Legal Text | 75 | **Merge into 2.1** (<⅓ page) |
| 3 Dataset Construction (intro) | 33 | Keep as orientation brief |
| 3.1 Source Documents | 82 | Keep |
| 3.2 Task Types | 152 | Keep |
| 3.3 Annotation Protocol | 229 | Trim — over-defends single-annotator choice |
| 3.4 Model-Based Validation | 94 | Keep |
| 3.5 Human IAA | 98 | Keep |
| 4.1 Models | 87 | Keep |
| 4.2 Prompting | 216 | Trim |
| 4.3 Scoring | 162 | **Expand** — this is now the paper's spine |
| 4.4 Human Reference Point | 79 | **Merge** with 5.2 |
| 5.1 Main Results | **46** | **Badly underweight** — a table with no interpretation |
| 5.2 Human Reference Comparison | 61 | Merge with 4.4; resize claim |
| 5.3 Significance & Tiers | 157 | Keep, trim repetition |
| 5.4 Task-Level Analysis | **45** | **Underweight** |
| 5.5 Difficulty Analysis | **43** | **Underweight** |
| 6 Error Taxonomy | 181 | Keep |
| 7 Discussion **and Conclusion** | 235 | **Split and expand** — payoff section is too thin |
| Limitations | 257 | Keep; longer than the Discussion, which is backwards |
| **NEW: Item discrimination** | — | **Add** — the approved second contribution |

**The core problem this map exposes:** the results sections total **195 words** across four
subsections. The paper presents tables and barely interprets them. In ACL two-column that reads as
terse; in TMLR single-column it will read as unfinished. This is where the added space goes.

## 5. Orphan check — confirmed defects

| Item | Problem | Fix |
|---|---|---|
| `tab:e1` (Table E1, few-shot) | **Never referenced in text** | Add a reference in the few-shot appendix |
| `app:prompt` | **Never referenced in text** | Reference it from §4.2 Prompting |

## 6. Repeated facts (Law 9 — state once, reference thereafter)

| Fact | Currently appears in | Keep it in |
|---|---|---|
| Llama 4 Scout 17B ≈ LLaMA-3.3-70B at ¼ params | abstract, §5.3, §7 (×2), conclusion | §5.3 once; reference later |
| GPT-OSS 120B ≈ 20B despite 6× size | abstract, §5.3, §7, conclusion | §5.3 once |
| DeepSeek R1 70B ranks 11th | abstract, §1, §6, §7, conclusion | abstract + §7 (it is the hero) |
| Gemini 2.5 Pro verbosity artifact | Table 2 caption, §5.1, §5.4, App. D | §4.3 once, as a scoring-regime example |

## 7. Where the 12 regenerated analyses go

| Analysis | Placement | Why |
|---|---|---|
| **IRT / item discrimination** (203 ceiling, 137 discriminative) | **Main text, new section** | Approved second contribution |
| **Kendall's W = 0.853** | Main text, §5.3 | Quantifies rank stability across tasks |
| **Error geometry / dissociation index** | Main text, §6 | Directly supports the strict-vs-corrected spine |
| Consensus-hard/easy (11 / 203) | Main text, with IRT | Independent cross-check of the ceiling count |
| Era stratification | Appendix | Supporting |
| Context-length nulls | Appendix | Honest null |
| Feature regression (flesch_ease 0.708) | Appendix | Supporting |
| TMP complexity nulls (12/12 n.s.) | Appendix | Honest null |
| CON balance | Already App. C | Keep |
| Scoring audit (100 items, 11 models) | Appendix, **scope stated** | Frozen at 11 models |
| RSTS, perturbation | Appendix **or cut**, scope stated | Frozen at 11 models |
| RAG (120 items, 1 model) | **Cut** | Too incomplete to report |

## 8. Length budget

Main body 3,024 words + ~2,000 new ≈ 5,000 words. In TMLR single-column 11pt with 6 figures and
~8 tables, that lands near **10–11 pages** — inside the 12-page fast-review track, with margin.
