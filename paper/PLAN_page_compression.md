# PLAN — compress main body 19pp -> 12pp

Created 2026-09-04. User explicitly chose "hard compress to 12pp" over shipping at 19pp
or a light trim. Resume by reading this file and continuing the first unchecked box.

## Ground rule

Relocate, don't delete. Every subsection moved to appendix keeps a compact pointer in the
main text (headline finding + "Appendix X"). Never cut a sentence that is the only place a
claim's evidence lives without also removing or hedging the claim itself (TMLR's #1
acceptance criterion: claims must be backed by evidence in the submission — an appendix
still counts, reviewers read it at their discretion).

Empirical tracking: word-count estimates are a rough guide only (LaTeX page count doesn't
scale linearly — headings, float placement, and table overhead matter). **Recompile and
check `PRE_SUBMISSION_AUDIT.py`'s page count after each numbered step**, not just at the end.

Baseline: 19pp main body, 11,305 words across draft_02/03/04/05/06/07/08/09.

## Result — 2026-09-04

19pp -> 17pp (2 pages, ~11%) via steps 1-11 below, all executed. Every safe/moderate cut and
relocation identified was applied; two full rounds run with empirical recompiles between them.
Second round (~180w of further trims: judge-pilot relocation, seed-noise tightening) moved 0
additional pages, confirming the remaining content is dense enough that further page gains would
require cutting into material that is the sole support for a claim still made in the main text
(Section 5's regime comparison, Section 6.5's judge validation, Section 6.6's significance
testing) -- exactly the stop condition step 12 below specifies. Stopped there per that condition;
reported to the user rather than cutting further unilaterally.

**Practical note surfaced by this result:** TMLR's fast-review threshold is binary at 12pp (<=12 ->
2wk review, >12 -> 4wk). 17pp is still on the same 4-week track as 19pp was -- the compression
improved the paper's density and cut real redundancy (verified via PRE_SUBMISSION_AUDIT.py, zero
regressions, zero broken refs) but did not cross the specific threshold. Getting to <=12pp would
need cutting into the paper's actual spine.

## Steps

- [x] **1. Dedup judge-regime definitions.** `draft_03_results_core.tex` opens Section 5 by
      re-defining strict/judge-only/judge-augmented almost verbatim to `draft_07`'s
      Section 4.3 Scoring (both explain the three regimes, "primary vs sensitivity," and why
      "corrected"/"semantic" are avoided). Keep the full definition in 4.3 (the methods
      home); compress Section 5's opening to a short recap + cross-reference.
      Est. save: ~135w.
- [x] **2. Merge discriminative-coverage restatement.** `draft_03` Section 6 explains D(k)
      and the 148/72/134 item counts three times: prose before Table `tab:effsize`, the
      table caption itself, and prose after the table. Keep the table as the source of
      truth; merge the two prose blocks into one interpretive pass that doesn't re-derive
      the counts the table already shows. Est. save: ~130w.
- [x] **3. Relocate Section 5.1 (Evaluation-Configuration Sensitivity) to the appendix.**
      ~600 words + Table `tab:matchedbudget`. This is a robustness/sensitivity check
      (analogous to judge-augmented already being demoted), not the paper's spine. Move the
      full text + table into Appendix G (G.1/G.2 already hold supporting detail). Main text
      keeps ~5 sentences: what was re-run, the headline shift range, the DeepSeek outlier
      flagged without a clean explanation, and the "measurement protocol, not just the
      model" tie-back. Est. save: ~525w + 1 table (~0.3-0.5pp on its own).
- [x] **4. Relocate Section 6.3 (Difficulty Stratification) detail to Appendix I.**
      Appendix I already holds the full table + all three difficulty figures
      (`tab:difficulty`, `fig:difficulty`, `fig:radar`, `fig:correlation`) — this section's
      job in the main text is just to state the headline finding (3/12 models decline
      monotonically) and point there. Cut the LLaMA-3.3-70B specific-example paragraph and
      tighten the interpretive paragraph; move both to Appendix I. Est. save: ~225w.
- [x] **5. Tighten the pre-registered-decision-rule paragraph** (`draft_08`, Section 6.5).
      Keep the honest disclosure (rule existed, didn't anticipate two regimes, both reported
      as co-equal) but cut the play-by-play. Est. save: ~60w.
- [x] **6. Trim the Section 6.6 forward-pointer** to Discussion's parameter-efficiency pairs
      (`draft_08`) — one sentence, naming the two comparisons without a "coming attractions"
      preview since Discussion covers them fully anyway. Est. save: ~25w.
- [x] **7. Tighten `draft_05` 6.4 Human Reference Point's "we state the limits plainly"
      paragraph** — keep the caveats, cut the elaboration. Est. save: ~40w.
- [x] **8. Convert Section 3.2 Task Types from four bolded prose paragraphs into a compact
      table** (Task | N | Definition | Example) — same information, tighter. Est. save:
      ~100w, plus reads cleaner.
- [x] **9. Compress Section 4.1's model-listing sentence.** The full provider/access-mode
      enumeration is already in Appendix G's model table (`tab:models`) — main text only
      needs "twelve models across major providers and access modes (Appendix G)". Est. save:
      ~55w.
- [x] **10. Trim Section 3.3 Annotation Protocol further** — cut the generic itemized
      review-criteria list (any benchmark paper could state it) and merge "Annotation scale
      rationale" into the surrounding prose rather than a separate labelled paragraph.
      Est. save: ~100w.
- [x] **11. Compress Section 4.2's literal system-prompt quote** — Appendix D already holds
      the "System Prompt Template." Main text describes the constraint in one sentence and
      points there instead of quoting it twice. Est. save: ~55w.
- [x] **12. Recompile, run `PRE_SUBMISSION_AUDIT.py`, record actual page count.** If still
      above 12pp, identify the next-safest relocation candidate (likely trimming
      `draft_08` 6.5's disagreement-breakdown prose or `draft_03`'s judge-augmented
      explanation, both currently fairly dense) and repeat. If 12pp is not reachable without
      cutting content that is the sole support for a claim still made in the main text,
      stop and report the actual page count reached plus what remains, rather than cutting
      further unilaterally.

## Verify (all must pass before reporting done)

- [x] `PRE_SUBMISSION_AUDIT.py` → PASS, 0 warnings on the page-count check specifically
      (or an honest report of the page count actually reached, per step 12's stop condition)
- [x] Every number that appears in a relocated block still appears exactly once, correctly,
      somewhere in the PDF (moved, not duplicated or dropped)
- [x] No orphaned cross-references (`\ref` to something that moved but wasn't relabeled)
- [x] Zero LaTeX errors, zero undefined references on recompile
