# PLAN — IndiaFinBench → TMLR

Created 2026-08-31. Resume by reading this file and starting at the first unchecked box.

## Context

EMNLP 2026 FinNLP desk-rejected V8 on 2026-08-29 for exceeding the 8-page limit. FinNLP's 2026
cycle is closed (both deadlines passed). Target is now **TMLR** — rules verified and recorded in
[TMLR_RULES.md](TMLR_RULES.md). Read that file before touching format.

**Key inversion:** TMLR has no page limit. Nothing needs cutting. Target ≤12pp main content to stay
on the 2-week review track. The paper has room to *grow*.

## Canonical inputs

> ⚠ **Source-of-truth resolved 2026-08-31.** Two lineages exist. The **LaTeX** branch is the
> upgraded, submitted, data-backed paper (human baseline 80.0%, 48/60). Every **.docx** — including
> the one named `..._Submitted.docx` and the newer Aug 6 copy — is the **pre-upgrade Word branch**
> carrying a superseded 69.0% / n=100 human baseline that no data file reproduces.
> **Never build from the docx.** Full evidence in `tmlr/NUMBERS.md` §0.

| Thing | Path |
|---|---|
| **Canonical source (LaTeX)** | `paper/tmlr/acl_source/acl_latex.tex` — from `C:\Users\Asus\Downloads\IndiaFinBench_ACL_LaTeX_final.zip` (dated 2026-07-05 = submission date; bundled preview PDF byte-size-identical to the submitted preview) |
| Figures + bib | `paper/tmlr/acl_source/figures/`, `indiafinbench.bib` |
| Submitted PDF | `...P05_IndiaFinBench__EMNLP-FinNLP__UNDER-REVIEW\P5_IndiaFinBench_V8_preview_EMNLP2026-FinNLP.pdf` (16pp) |
| ~~V8 docx~~ | **superseded — pre-upgrade Word branch, do not use** |
| Results, 12 models × 406 items | `evaluation/results/*.csv` |
| Judge-corrected | `evaluation/results_judged/` |
| Unused analyses | `evaluation/novel_methods/` (12 dirs) |
| Generators | `scripts/exp1_*.py` … `exp11_*.py`, `run_all_experiments.py` |
| Build dir (new) | `paper/tmlr/` |

**Open question:** `D:\Projects\IndiaFinBench\IndiaFinBench\` is a nested duplicate tree with its own
`scripts/` and `paper/`. Determine which is canonical before running anything. Do not edit both.

## Acceptance criteria

1. Zero format desk-reject risk: TMLR stylefile, ≤12pp main content, anonymized, checklist ticked.
2. Every number traces to a results file; no 11-vs-12-model mismatch anywhere.
3. Claims sized to evidence — especially the human-comparison claim in the abstract.
4. Reframed so the measurement finding leads and the benchmark is the instrument.
5. Reads as human-written; no LLM-tell prose.

---

## Phase A — Ground truth (free, local compute only)

The `novel_methods` outputs cover **11 models**; the paper reports **12** (missing Gemini 2.5 Pro).
Nothing gets written into the paper until this is reconciled.

- [x] **A0** Canonical tree is the **outer** `D:\Projects\IndiaFinBench\`. The nested
      `IndiaFinBench\` is a gitignored export snapshot (`.gitignore:68`), holds only 13 result files,
      lacks `gemini25_pro_results.csv`, and has identical script hashes. **Never edit or run it.**
- [x] **A1** No hardcoded 11-model list exists. `scripts/novel_methods_utils.py:26` `MODEL_FILES`
      already names all 12; `load_all_results()` silently skips missing files, which is why earlier
      runs produced 11-model outputs. Re-running alone fixes it.
      **API split verified by call site:** free = exp1, exp3, exp5, exp6, exp7, exp8, exp11.
      Paid = exp2 (`:133`), exp4 (`:102`), exp9 (`:158,165`), exp10 (`:67,69`).
- [x] **A2** Re-ran all 7 free experiments, every one exit 0, all now covering 12 models.
      Originals backed up to `evaluation/_novel_methods_BACKUP_11models_2026-08-31/`.
      Eight summary values changed — see the change table in `NUMBERS.md`.
- [x] **A3** `paper/tmlr/NUMBERS.md` written; every row carries a real path.
- [~] **A4** Core results table **fully verified** — all 60 cells reproduce the submitted V8 exactly,
      and the 70.4–89.7 abstract range holds. Remaining items (judge-corrected bounds, bootstrap
      p-values, Wilson CIs, IAA, dataset composition) listed as a checklist in `NUMBERS.md` §4.

## Phase B — Restructure and reframe (skill: manuscript-flow)

- [x] **B1** Paper map written to `tmlr/STRUCTURE.md` from measured section lengths
      (`tmlr/analyze_tex.py`). Main body is only **3,024 words**; the four results subsections total
      **195 words**. Two orphan floats found (`tab:e1`, `app:prompt`). Four facts repeated 4–5×.
- [x] **B3a** Title + abstract drafted → `tmlr/draft_01_title_abstract.tex`
- [x] **B2a** Introduction drafted on the measurement spine → `tmlr/draft_02_introduction.tex`
- [x] **B4** Core results drafted → `tmlr/draft_03_results_core.tex` (scoring regime + effective size)
- [ ] **B5** ⚠ **Terminology correction required repo-wide.** `scripts/exp7_item_discrimination.py`
      is labelled "IRT-lite" and writes to `evaluation/novel_methods/iRT_analysis/`, but it fits **no
      item-response model**. It computes $s = 4p(1-p)$, a normalised variance over the 12-model
      panel. The paper must never call this IRT. Drafts already use "spread"/"effective size".
      Consider renaming the output directory to `item_spread_analysis/`.
- [x] **B2** Structure decided — see `tmlr/STRUCTURE.md`. Decide the TMLR structure. Measurement finding leads; India benchmark is the instrument.
      Decide for each of the 12 unused analyses: main text, appendix, or cut. Justify each.
      *Verify:* structure table in `paper/tmlr/STRUCTURE.md` with a rationale per section.
- [x] **B3** Claim-sizing done — human claim resized in abstract + §human. Claim-sizing pass. The abstract currently asserts LLMs "match, not exceed, careful human
      judgment" on the strength of one non-specialist evaluator over 60 items. Rewrite to the
      evidence's actual strength. Sweep for every other claim that outruns its support.
      *Verify:* each abstract claim annotated with the evidence backing it.
- [ ] **B4** Own the ceiling-effect finding: IRT says 215/406 items are ceiling items. Decide where
      this goes and how it is framed (rigor signal, not buried weakness).

## Phase C — Build in TMLR LaTeX

- [ ] **C1** Download the TMLR stylefile; scaffold `paper/tmlr/` and confirm a stub compiles.
      *Verify:* stub PDF builds with no errors.
- [ ] **C2** Extract V8's text (docx skill) into LaTeX section by section.
- [ ] **C3** Integrate the Phase-B-selected new analyses with numbers from `NUMBERS.md`.
- [ ] **C4** Figures: regenerate at 300dpi, serif, colorblind-safe, via the existing generator
      scripts so each figure remains reproducible from a results file.
      *Verify:* every figure has a regenerating script; no figure is a pasted bitmap.

## Phase D — Humanize (skill: humanizer)

- [ ] **D1** De-LLM pass over all new and rewritten prose. Vary sentence length, concrete verbs, no
      symmetrical triads, no "moreover" cascades, no hedging filler.
- [ ] **D2** Voice-match against V8's existing prose so new sections don't read as bolted on.

## Phase E — Verify before submitting (skill: verification-before-completion)

- [ ] **E1** Compile clean; no missing refs, no overfull boxes that break the layout.
- [ ] **E2** Main content ≤12pp before references. Count and state the number.
- [ ] **E3** Anonymization sweep: paper, supplementary, and the linked repo. The 4open.science link
      already in V8 must stay anonymous.
- [ ] **E4** Re-scan every number at every occurrence (abstract, body, tables, figures, appendix)
      for drift.
- [ ] **E5** **Originality check** — TMLR bars reuse of text/figures/results with anything submitted
      in parallel at an archival venue. P13 (ICLR 2027, "Benchmark Accuracy Is Not an Identified
      Quantity") and P12 (TAE@NeurIPS) are in the same intellectual family. Diff them against this
      paper. If P13 is live at ICLR and shares results, resolve before submitting.
      *Verify:* explicit written finding on overlap.
- [ ] **E6** Confirm TMLR quota balance on the official calculator.
- [ ] **E7** Nominate an Action Editor.
- [ ] **E8** Walk the full pre-submission checklist in `TMLR_RULES.md`.

## Housekeeping

- [ ] Update `_INDEX.md`: P05 is desk-rejected, not "Under review · notify 09-01".
- [ ] Update the manuscript-pipeline memory file with the same correction.

## Cost constraint

**No paid API calls.** User has ~$1 of credits. Everything above is post-hoc analysis over stored
predictions, local compute, or writing.

Ruled out as not-free (verified 2026-08-31):
- Completing haiku (150/406), nemotron-120B (140/406), qwen3-235B (120/406) — needs paid inference.
- Completing the RAG evaluation on API models (`rag_results_partial.csv` = 120 items, 1 model).
- *Possible free variant:* RAG on the three local Ollama models (llama3:8b, mistral:7b, gemma3:4b)
  on the RTX 4060. Time-expensive; decide in Phase B whether it earns its place.


---

## Phase B COMPLETE — 2026-08-31

Six drafts in `tmlr/`, 4,590 words:

| File | Words | Contents |
|---|---|---|
| `draft_01_title_abstract.tex` | 354 | New title + abstract, hero result leads |
| `draft_02_introduction.tex` | 794 | Measurement spine, 6 contributions |
| `draft_03_results_core.tex` | 1194 | Scoring-regime table + effective size |
| `draft_04_related_work.tex` | 392 | 2.1+2.2 merged; adds scoring-conventions gap |
| `draft_05_results_detail.tex` | 970 | Replaces the 195-word results problem |
| `draft_06_discussion_conclusion.tex` | 886 | Split from the merged 235-word section |

Carried over unchanged from `acl_source/acl_latex.tex`: Dataset Construction, Experimental Setup,
Error Analysis, Limitations, Ethics, appendices.

### Stale artifacts fixed this phase
- `scripts/generate_figures.py` — two hardcoded 11-model lists (`MODEL_FILES:52`, `short_names:335`)
  omitted Gemini 2.5 Pro. Fixed; backup `.bak`. Bootstrap now 12 models / **66 pairs**.
- Figures regenerated at 12 models.

### Phase C entry point
1. **Check `acl_source/figures/*.png` (2026-07-03) for 11 vs 12 models.** If 11, the submitted
   paper's figures contradict its own text. Regenerate from the fixed script and map filenames.
2. Download TMLR stylefile; scaffold `tmlr/build/`; confirm a stub compiles.
3. Assemble drafts + carried-over sections in canonical order.
4. Apply the §5 number corrections (41 pairs; p=0.793 / 0.907 / 0.057; 10 of 12 CON).
5. Fix orphans `tab:e1`, `app:prompt`.
